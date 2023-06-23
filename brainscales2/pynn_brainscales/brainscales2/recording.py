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


class MADCRecorderSetting(NamedTuple):
    cell_id: int
    comp_id: halco.CompartmentOnLogicalNeuron
    readout_source: hal.NeuronConfig.ReadoutSource


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator
    madc_variables = ["v", "exc_synin", "inh_synin", "adaptation"]

    def __init__(self, population, file=None):
        super().__init__(population, file=file)
        self.changed_since_last_run = True

    def _configure_madc(self, variables: Set[str],
                        recording_sites: Set[RecordingSite]):
        if len(variables) == 0 or len(recording_sites) == 0:
            return
        if len(variables) > 1:
            raise ValueError("Can only set 1 analog record type per chip.")
        if len(recording_sites) > 1:
            raise ValueError("Can only record single neuron/location via "
                             "MADC.")

        variable = next(iter(variables))
        assert variable in Recorder.madc_variables
        recording_site = next(iter(recording_sites))

        readout_source = None
        if variable == "v":
            readout_source = hal.NeuronConfig.ReadoutSource.membrane
        elif variable == "exc_synin":
            readout_source = hal.NeuronConfig.ReadoutSource.exc_synin
        elif variable == "inh_synin":
            readout_source = hal.NeuronConfig.ReadoutSource.inh_synin
        elif variable == "adaptation":
            readout_source = hal.NeuronConfig.ReadoutSource.adaptation
        else:
            raise RuntimeError("Encountered not handled MADC case.")

        madc_recorder = MADCRecorderSetting(cell_id=recording_site.cell_id,
                                            comp_id=recording_site.comp_id,
                                            readout_source=readout_source)

        # check if MADC recorder already set. Ignore if already
        # existing config is set again.
        global_madc_rec = self._simulator.state.madc_recorder
        if global_madc_rec is not None and (global_madc_rec != madc_recorder):
            raise ValueError(
                f"Analog record for ID {global_madc_rec.cell_id} of "
                f"type {global_madc_rec.readout_source} at compartment "
                f"{global_madc_rec.comp_id} is already active. Only one "
                "concurrent analog readout supported.")
        self._simulator.state.madc_recorder = madc_recorder

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
                self._simulator.state.madc_recorder = None

    def _clear_simulator(self):
        self._simulator.state.spikes = []
        self._simulator.state.times = []
        self._simulator.state.madc_samples = []

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
    # pylint: disable=too-many-locals,consider-using-f-string
    def _get_current_segment(
            self,
            filter_ids=None,
            variables='all',
            clear=False):
        segment = neo.Segment(
            name="segment%03d" % self._simulator.state.segment_counter,
            description=self.population.describe(),
            rec_datetime=datetime.now())
        variables_to_include = set(self.recorded.keys())
        if variables != 'all':
            variables_to_include = variables_to_include.\
                intersection(set(variables))
        for variable in variables_to_include:
            if variable == 'spikes':
                t_stop = self._simulator.state.t * pq.ms
                sids = sorted(self.filter_recorded('spikes', filter_ids))
                data = self._get_spiketimes(sids, clear=clear)

                segment.spiketrains = []
                for identifier in sids:
                    times = pq.Quantity(data.get(identifier, []), pq.ms)
                    if times.size > 0 and times.max() > t_stop:
                        warn("Recorded at least one spike after t_stop")
                        times = times[times <= t_stop]
                    location = self._get_location_label(identifier.comp_id)
                    segment.spiketrains.append(
                        neo.SpikeTrain(
                            times,
                            t_start=self._recording_start_time,
                            t_stop=t_stop,
                            units='ms',
                            source_population=self.population.label,
                            source_id=int(identifier.cell_id),
                            source_location=location,
                            source_compartment=int(identifier.comp_id),
                            source_index=self.population.id_to_index(
                                int(identifier.cell_id)))
                    )
            else:
                if variable not in Recorder.madc_variables:
                    raise ValueError(
                        "Only implemented for membrane potential 'v' and "
                        + "technical parameters: '{exc,inh}_synin', "
                        + "'adaptation'.")
                ids = sorted(self.filter_recorded(variable, filter_ids))
                if not ids:
                    # don't add a signal when no ids of the requested selection
                    # were recorded
                    continue
                assert len(ids) == 1
                if simulator.state.times.size > 0:
                    units = self.population.find_units(variable)
                    index_array = np.array(
                        [self.population.id_to_index(myid.cell_id) for myid in
                         ids])
                    location = self._get_location_label(ids[0].comp_id)
                    signal = neo.IrregularlySampledSignal(
                        times=simulator.state.times,
                        signal=simulator.state.madc_samples,
                        units=units,
                        time_units='ms',
                        name=variable,
                        source_population=self.population.label,
                        source_ids=np.array([ids[0].cell_id]),
                        source_locations=np.array([location]),
                        source_compartments=np.array([int(ids[0].comp_id)]),
                        array_annotations={"channel_index": index_array})
                    segment.irregularlysampledsignals.append(signal)  # pylint: disable=no-member
        return segment

    def _local_count(self, variable, filter_ids):
        counts = {}
        if variable == "spikes":
            for filter_id in self.filter_recorded(variable, filter_ids):
                counts[int(filter_id)] = len(simulator.state.spikes)
        else:
            raise ValueError("Only implemented for spikes")
        return counts
