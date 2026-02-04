from itertools import product, compress
from typing import Sequence, Set, List, Optional, Tuple
from datetime import datetime
from warnings import warn
import numpy as np
import neo
import quantities as pq
import pyNN.recording
import pyNN.errors

from pyNN.recording import gather_blocks, filter_by_variables, \
    remove_duplicate_spiketrains

from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.recording_data import RecordingSite, \
    RecordingConfig, GrenadeRecId, RecordingType
from dlens_vx_v3 import halco


def normalize_variables_arg(variables):
    """If variables is a single string, encapsulate it in a list."""
    if isinstance(variables, str) and variables != 'all':
        return [variables]
    return variables


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator

    def __init__(self, population, file=None):
        super().__init__(population, file=file)
        self.changed_since_last_run = True

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
               locations=None, *, device="madc"):
        self.changed_since_last_run = True
        # MADC based recording is only possible for one neuron on a chip
        # it is therefore checked for population size one and no multi
        # assignment before the parent record function is called which modifies
        # state
        if sampling_interval:
            raise NotImplementedError("Sampling interval not implemented.")
        variable_list = normalize_variables_arg(variables)

        ids = {id for id in ids if id.local}
        recording_sites = self._get_recording_sites(ids, locations)
        grenade_ids = {self._rec_site_to_grenade_index(rec_site) for rec_site
                       in recording_sites}

        if device == "madc":
            self._simulator.state.recordings[-1].config.add_madc_recording(
                set(variable_list).intersection(
                    RecordingConfig.analog_observable_names),
                grenade_ids)
        elif 'pad' in device:
            pad, buffered = self._device_name_to_pad_config(device)
            self._simulator.state.recordings[-1].config.add_pad_readout(
                set(variable_list).intersection(
                    RecordingConfig.analog_observable_names),
                grenade_ids,
                pad=pad, buffered=buffered)
        else:
            raise ValueError(f'Device "{device}" is not supported')

        self._simulator.state.log.debug(f'Recorder.record(<{len(ids)} cells>)')
        for variable in variable_list:
            if not self.population.can_record(variable):
                raise pyNN.errors.RecordingError(variable,
                                                 self.population.celltype)
            # save all recording sites in a set
            self.recorded[variable] |= recording_sites

    @staticmethod
    def _device_name_to_pad_config(device: str) -> Tuple[str, bool]:
        '''
        Extract pad and buffering mode from device name.

        The device name is of the form 'pad_[0|1][|_unbuffered|_buffered]'.
        Raise ValueError if device name does not fit format. Otherwise,
        extract pad as well as buffering mode from device name.
        :param device: Device to record.
        :returns: Pad, buffering enabled.
        '''
        parts = device.split("_")
        correct_start = parts[0] == 'pad'
        correct_pad_number = len(parts) > 1 and parts[1] in ['0', '1']
        correct_buffering = len(parts) < 3 or parts[2] in ['buffered',
                                                           'unbuffered']
        if len(parts) > 3 or not (correct_start and correct_pad_number
                                  and correct_buffering):
            raise ValueError(f'Device "{device}" is not supported')
        pad = int(parts[1])
        unbuffered = 'unbuffered' in device

        return pad, not unbuffered

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

        for variable in self.recorded:
            for rec_site in self.recorded[variable]:
                grenade_id = self._rec_site_to_grenade_index(rec_site)
                # this removes the recording site from the configuration as
                # well as previously recorded data since the data can no longer
                # be retrieved.
                self._simulator.state.recordings[-1].remove(
                    recording_site=grenade_id)

    def _clear_simulator(self):
        # here we only remove the recorded data but keep the configuration
        for variable in self.recorded:
            for rec_site in self.recorded[variable]:
                grenade_id = self._rec_site_to_grenade_index(rec_site)
                for recording in self._simulator.state.recordings:
                    recording.data.remove(
                        recording_site=grenade_id)

    def _get_spiketimes(self, ids, snippet_idx, clear=False):
        """Returns a dict containing the recording site and its spiketimes."""
        all_spiketimes = {}
        recording = self._simulator.state.recordings[snippet_idx]
        for rec_site in ids:
            grenade_id = self._rec_site_to_grenade_index(rec_site)
            if grenade_id in recording.data.spikes:
                all_spiketimes[rec_site] = recording.data.spikes[grenade_id]
            if clear:
                recording.data.remove(recording_site=grenade_id,
                                      recording_type=RecordingType.SPIKES)
        return all_spiketimes

    def _rec_site_to_grenade_index(self, rec_site: RecordingSite
                                   ) -> GrenadeRecId:
        index_on_pop = np.where(
            self.population.all_cells == int(rec_site.cell_id))[0]
        assert len(index_on_pop) == 1
        return GrenadeRecId(simulator.state.populations.index(self.population),
                            index_on_pop[0], int(rec_site.comp_id))

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
    # pylint: disable=too-many-locals,consider-using-f-string,line-too-long,invalid-name,unreachable,no-member,no-else-raise,too-many-branches,too-many-statements,redefined-builtin,too-many-nested-blocks
    def _get_current_segment(self, filter_ids=None, variables='all', clear=False):
        segment = neo.Segment(name="segment%03d" % self._simulator.state.segment_counter,
                              description=self.population.describe(),
                              # would be nice to get the time at the start of the recording,
                              # not the end
                              rec_datetime=datetime.now())
        variables_to_include = set(self.recorded.keys())
        if variables != 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        for snippet_idx in range(len(self._simulator.state.recordings) - 1):
            t_start = sum(self._simulator.state.runtimes[0:snippet_idx]) * pq.ms
            t_stop = t_start + self._simulator.state.runtimes[snippet_idx] * pq.ms  # must run on all MPI nodes
            for variable in sorted(variables_to_include):
                if variable == 'spikes':
                    sids = sorted(self.filter_recorded('spikes', filter_ids))
                    data = self._get_spiketimes(sids, snippet_idx, clear=clear)

                    if isinstance(data, dict):
                        for id, spikes in data.items():
                            times = pq.Quantity(spikes, pq.ms)
                            times += t_start
                            if times.size > 0 and times.max() > t_stop:
                                warn("Recorded at least one spike after t_stop")
                                times = times[times <= t_stop]
                            location = self._get_location_label(id.comp_id)
                            segment.spiketrains.append(
                                neo.SpikeTrain(
                                    times,
                                    t_start=t_start,
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
                        id_array, times = data  # pylint: disable=unbalanced-dict-unpacking
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
                    signal_array, times_array = self._get_all_signals(variable, ids, snippet_idx, clear=clear)
                    times_array += t_start
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

    def _get_all_signals(self, variable, ids, snippet_idx, clear=False):
        times = []
        values = []
        recording = self._simulator.state.recordings[snippet_idx]
        for id in ids:
            if id not in self.recorded.get(variable, set()):
                raise RuntimeError("No samples were recorded for population "
                                   f"'{self.population.label}' at recording "
                                   f"{id}.")
            grenade_id = self._rec_site_to_grenade_index(id)
            if grenade_id not in recording.config.analog_observables:
                values.append([])
                times.append([])
                continue

            recorded_var = recording.config.analog_observables[grenade_id]
            if recorded_var != RecordingConfig.str_to_source_map.get(variable):
                raise RuntimeError(f"'{recorded_var}' was recorded but "
                                   f"'{variable}' was requested "
                                   f"(population: {self.population.label}, "
                                   f"recording_site: {id}).")
            if grenade_id not in recording.data.madc:
                # no samples have been recorded yet
                continue

            madc_recording = recording.data.madc[grenade_id]
            times.append(madc_recording.times)
            values.append(madc_recording.values)
            if clear:
                recording.data.remove(recording_site=grenade_id,
                                      recording_type=RecordingType.MADC)

        return np.array(values, dtype=object), np.array(times, dtype=object)

    def _local_count(self, variable, filter_ids):
        """
        Count number of spikes for the given filter_ids.
        """
        if variable != "spikes":
            raise ValueError("Only implemented for spikes")
        counts = {}
        for filter_id in filter_ids:
            count = 0
            for recording_site in self.filter_recorded(variable,
                                                       filter_ids):
                for snippet_idx in \
                        range(len(self._simulator.state.recordings) - 1):
                    spikes = self._get_spiketimes(recording_site,
                                                  snippet_idx, clear=False)
                    count += len(spikes)
            counts[int(filter_id)] = len(count)
        return counts

    # TODO: align handling of varibales to PyNN 0.12
    # this function was copied from PyNN 0.10.1 and the agruments were
    # adjusted for PyNN 0.12
    # pylint: disable=too-many-arguments
    def get(self, variables, gather=False, filter_ids=None, clear=False,
            annotations=None, locations=None):
        """Return the recorded data as a Neo `Block`."""
        del locations  # we do not yet support PyNN 0.12
        variables = normalize_variables_arg(variables)
        data = neo.Block()
        data.segments = [filter_by_variables(segment, variables)
                         for segment in self.cache]
        if self._simulator.state.running:
            # reset() has not been called, so current segment is not in cache
            data.segments.append(self._get_current_segment(
                filter_ids=filter_ids, variables=variables, clear=clear))
        for segment in data.segments:
            segment.block = data
        data.name = self.population.label
        data.description = self.population.describe()
        data.rec_datetime = data.segments[0].rec_datetime
        data.annotate(**self.metadata)
        if annotations:
            data.annotate(**annotations)
        if gather and self._simulator.state.num_processes > 1:
            data = gather_blocks(data)
            if (
                hasattr(self.population.celltype, "always_local")
                and self.population.celltype.always_local
            ):
                data = remove_duplicate_spiketrains(data)
        if clear:
            self.clear()
        return data
