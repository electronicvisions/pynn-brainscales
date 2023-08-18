from itertools import product, compress
from typing import NamedTuple, Sequence, Set, List, Optional
from datetime import datetime
from warnings import warn
import numpy as np
import neo
import quantities as pq
import pyNN.recording
import pyNN.errors

from pynn_brainscales.brainscales2 import simulator
from dlens_vx_v3 import hal, halco


class RecordingSite(NamedTuple):
    cell_id: int  # Type pynn.simulator.ID (cyclic import)
    comp_id: halco.CompartmentOnLogicalNeuron


class MADCRecordingSite(NamedTuple):
    population: int
    neuron_on_population: int
    compartment_on_neuron: halco.CompartmentOnLogicalNeuron


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator
    _var_name_to_readout_source = \
        {"v": hal.NeuronConfig.ReadoutSource.membrane,
         "exc_synin": hal.NeuronConfig.ReadoutSource.exc_synin,
         "inh_synin": hal.NeuronConfig.ReadoutSource.inh_synin,
         "adaptation": hal.NeuronConfig.ReadoutSource.adaptation}
    madc_variables = list(_var_name_to_readout_source)

    def __init__(self, population, file=None):
        super().__init__(population, file=file)
        self.changed_since_last_run = True

    # pylint: disable=too-many-branches
    def _configure_madc(self, variables: Set[str],
                        recording_sites: Set[RecordingSite]):
        if len(variables) == 0 or len(recording_sites) == 0:
            return
        if len(variables) > 1:
            raise ValueError("Can only set 1 analog record type per neuron.")
        if len(recording_sites) > 2:
            raise ValueError("Can only record at most two neurons/locations "
                             "via MADC.")

        variable = next(iter(variables))
        assert variable in Recorder.madc_variables
        readout_source = self._var_name_to_readout_source[variable]

        for recording_site in recording_sites:
            neuron_on_population = int(np.where(
                self.population.all_cells
                == int(recording_site.cell_id))[0][0])

            new_rec_site = MADCRecordingSite(
                population=self._simulator.state.populations.index(
                    self.population),
                neuron_on_population=neuron_on_population,
                compartment_on_neuron=recording_site.comp_id)

            # check if MADC recording already enabled for this site.
            all_rec_sites = self._simulator.state.madc_recording_sites
            if new_rec_site in all_rec_sites:
                if all_rec_sites[new_rec_site] == readout_source:
                    # recording already set.
                    return
                raise ValueError("Only one source can be recorded per neuron.")

            # MADC recording not enabled for this site.
            if len(all_rec_sites) >= 2:
                raise ValueError(
                    "Can only record at most two neurons/locations via MADC")
            all_rec_sites[new_rec_site] = readout_source

    def _get_recording_sites(self, neuron_ids: List[int],
                             locations: Optional[Sequence[str]] = None
                             ) -> Set[RecordingSite]:
        if locations is None:
            # record first compartment if locations is not defined
            comp_ids = [halco.CompartmentOnLogicalNeuron()]
        else:
            celltype = self.population.celltype
            try:
                comp_ids = celltype.get_compartment_ids(locations)
            except AttributeError as err:
                raise RuntimeError('Can not extract recording locations for '
                                   f'celltype "{celltype.__name__}".') from err

        return {RecordingSite(n_id, c_id) for n_id, c_id in
                product(neuron_ids, comp_ids)}

    def record(self, variables, ids, sampling_interval=None,
               locations=None):
        self.changed_since_last_run = True
        # MADC based recording is only possible for one neuron on a chip
        # it is therefore checked for population size one and no multi
        # assignment before the parent record function is called which modifies
        # state
        if sampling_interval:
            raise NotImplementedError("Sampling interval not implemented.")
        variable_list = pyNN.recording.normalize_variables_arg(variables)

        madc_variables = set(variable_list).intersection(self.madc_variables)
        ids = {id for id in ids if id.local}
        recording_sites = self._get_recording_sites(ids, locations)
        self._configure_madc(madc_variables, recording_sites)

        self._simulator.state.log.debug(f'Recorder.record(<{len(ids)} cells>)')

        for variable in variable_list:
            if not self.population.can_record(variable):
                raise pyNN.errors.RecordingError(variable,
                                                 self.population.celltype)
            # save all recording sites in a set
            self.recorded[variable] |= recording_sites

    # Overwrite base function since `self.recorded` saves sets of
    # RecordingSite. `filter_ids` filters for cell_ids therefore, we have to
    # implement a different filter function
    def filter_recorded(self, variable, filter_ids):
        if filter_ids is None:
            return self.recorded[variable]
        recording_sites = list(self.recorded[variable])
        cell_ids = np.array([x[0] for x in recording_sites])
        return set(compress(recording_sites, np.isin(cell_ids, filter_ids)))

    def _record(self, variable, new_ids, sampling_interval=None):
        pass

    def _reset(self):
        self.changed_since_last_run = True
        # only MADC record setting needs to be reset for BSS back end. As it's
        # a global state we check if a record parameters is MADC based
        for variable in self.recorded:
            if variable in Recorder.madc_variables:
                self._simulator.state.madc_recording_sites = {}

    def _clear_simulator(self):
        self._simulator.state.spikes = []
        self._simulator.state.madc_recordings = {}

    # pylint: disable=unused-argument
    def _get_spiketimes(self, ids, clear=None):
        """Returns a dict containing the recording site and its spiketimes."""
        all_spiketimes = {}
        for rec_site in ids:
            index_on_pop = np.where(
                self.population.all_cells == int(rec_site.cell_id))[0]
            assert len(index_on_pop) == 1
            neuron_idx = (simulator.state.populations.index(self.population),
                          index_on_pop[0], int(rec_site.comp_id))
            if neuron_idx in simulator.state.spikes:
                all_spiketimes[rec_site] = simulator.state.spikes[neuron_idx]
        return all_spiketimes

    def _get_location_label(self, comp_id: halco.CompartmentOnLogicalNeuron
                            ) -> str:
        try:
            return self.population.celltype.get_label(comp_id)
        except AttributeError:
            return 'unlabeled'

    # Patch to support multi-compartmental neuron models and to allow recording
    # observables at different locations.
    # Specifically, we add the location infomation as well the compartment id
    # as an annotation to recorded spike trains and analog signals.
    # Furthermore, we return `RecordingSite` in `filter_recorded()` which
    # consist of the cell id as well as comaprtment id. This has to be handled.
    # pylint: disable=too-many-locals,consider-using-f-string,line-too-long,invalid-name,unreachable,no-member,no-else-raise,too-many-branches,too-many-statements,redefined-builtin
    def _get_current_segment(self, filter_ids=None, variables='all', clear=False):
        segment = neo.Segment(name="segment%03d" % self._simulator.state.segment_counter,
                              description=self.population.describe(),
                              # would be nice to get the time at the start of the recording,
                              # not the end
                              rec_datetime=datetime.now())
        variables_to_include = set(self.recorded.keys())
        if variables != 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        for variable in variables_to_include:
            if variable == 'spikes':
                t_stop = self._simulator.state.t * pq.ms  # must run on all MPI nodes
                sids = sorted(self.filter_recorded('spikes', filter_ids))
                data = self._get_spiketimes(sids, clear=clear)

                if isinstance(data, dict):
                    for id in sids:
                        times = pq.Quantity(data.get(id, []), pq.ms)
                        if times.size > 0 and times.max() > t_stop:
                            warn("Recorded at least one spike after t_stop")
                            times = times[times <= t_stop]
                        location = self._get_location_label(id.comp_id)
                        segment.spiketrains.append(
                            neo.SpikeTrain(
                                times,
                                t_start=self._recording_start_time,
                                t_stop=t_stop,
                                units='ms',
                                source_population=self.population.label,
                                source_id=int(id.cell_id),
                                source_location=location,
                                source_compartment=int(id.comp_id),
                                source_index=self.population.id_to_index(int(id.cell_id)))
                        )
                        for train in segment.spiketrains:
                            train.segment = segment
                else:
                    raise RuntimeError("This code path is not needed for "
                                       "BSS-2 and should not be reached.")

                    assert isinstance(data, tuple)
                    id_array, times = data
                    times *= pq.ms
                    if times.size > 0 and times.max() > t_stop:
                        warn("Recorded at least one spike after t_stop")
                        mask = times <= t_stop
                        times = times[mask]
                        id_array = id_array[mask]
                    segment.spiketrains = neo.spiketrainlist.SpikeTrainList.from_spike_time_array(
                        times, id_array,
                        np.array(sids, dtype=int),
                        t_stop=t_stop,
                        units="ms",
                        t_start=self._recording_start_time,
                        source_population=self.population.label
                    )
                    segment.spiketrains.segment = segment
            else:
                ids = sorted(self.filter_recorded(variable, filter_ids))
                signal_array, times_array = self._get_all_signals(variable, ids, clear=clear)
                mpi_node = self._simulator.state.mpi_rank  # for debugging
                if signal_array.size > 0:
                    # may be empty if none of the recorded cells are on this MPI node
                    units = self.population.find_units(variable)
                    source_ids = np.array([int(id.cell_id) for id in ids])
                    channel_index = np.array([self.population.id_to_index(id.cell_id) for id in ids])
                    if self.record_times:
                        if signal_array.shape == times_array.shape:
                            # in the current version of Neo, all channels in
                            # IrregularlySampledSignal must have the same sample times,
                            # so we need to create here a list of signals
                            signals = [
                                neo.IrregularlySampledSignal(
                                    np.array(times_array[i], dtype=np.float32),
                                    np.array(signal_array[i], dtype=np.float32),
                                    units=units,
                                    time_units=pq.ms,
                                    name=variable,
                                    source_ids=[int(cell_id.cell_id)],
                                    source_locations=[self._get_location_label(cell_id.comp_id)],
                                    source_population=self.population.label,
                                    source_compartments=[int(cell_id.comp_id)],
                                    array_annotations={"channel_index": [self.population.id_to_index(cell_id.cell_id)]}
                                )
                                for i, cell_id in enumerate(ids)
                            ]
                        else:
                            raise RuntimeError("This code path is not needed for "
                                               "BSS-2 and should not be reached.")

                            # all channels have the same sample times
                            assert signal_array.shape[0] == times_array.size
                            signals = [
                                neo.IrregularlySampledSignal(
                                    times_array, signal_array, units=units, time_units=pq.ms,
                                    name=variable, source_ids=source_ids,
                                    source_population=self.population.label,
                                    source_locations=[self._get_location_label(cell_id.comp_id) for cell_id in ids],
                                    source_compartments=[int(cell_id.comp_id) for cell_id in ids],
                                    array_annotations={"channel_index": channel_index}
                                )
                            ]
                        segment.irregularlysampledsignals.extend(signals)
                        for signal in signals:
                            signal.segment = segment
                    else:
                        raise RuntimeError("This code path is not needed for "
                                           "BSS-2 and should not be reached.")

                        t_start = self._recording_start_time
                        t_stop = self._simulator.state.t * pq.ms
                        sampling_period = self.sampling_interval * pq.ms
                        current_time = self._simulator.state.t * pq.ms
                        signal = neo.AnalogSignal(
                            signal_array,
                            units=units,
                            t_start=t_start,
                            sampling_period=sampling_period,
                            name=variable, source_ids=source_ids,
                            source_population=self.population.label,
                            array_annotations={"channel_index": channel_index}
                        )
                        assert signal.t_stop - current_time - 2 * sampling_period < 1e-10
                        self._simulator.state.log.debug(
                            "%d **** ids=%s, channels=%s", mpi_node,
                            source_ids, signal.array_annotations["channel_index"])
                        segment.analogsignals.append(signal)
                        signal.segment = segment
        return segment

    def _get_all_signals(self, variable, ids, clear=None):
        del clear  # not implemented
        assert len(ids) <= 2
        if len(ids) == 0:
            return np.array([]), np.array([])
        if variable not in Recorder.madc_variables:
            raise ValueError("Only implemented for membrane potential 'v' and"
                             + "technical parameters: '{exc,inh}_synin', "
                             + "'adaptation'.")
        times = []
        values = []
        for madc_recording in self._simulator.state.madc_recordings.values():
            times.append(madc_recording.times)
            values.append(madc_recording.values)
        return np.array(values, dtype=object), np.array(times, dtype=object)

    def _local_count(self, variable, filter_ids):
        counts = {}
        if variable == "spikes":
            for filter_id in self.filter_recorded(variable, filter_ids):
                counts[int(filter_id)] = len(simulator.state.spikes)
        else:
            raise ValueError("Only implemented for spikes")
        return counts
