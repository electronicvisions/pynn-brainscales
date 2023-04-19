from typing import NamedTuple
from datetime import datetime
from warnings import warn
import numpy as np
import neo
import quantities as pq
import pyNN.recording
import pyNN.errors
from pynn_brainscales.brainscales2 import simulator
from dlens_vx_v3 import hal, halco


class MADCRecorderSetting(NamedTuple):
    cell_id: int
    readout_source: hal.NeuronConfig.ReadoutSource


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator
    madc_variables = ["v", "exc_synin", "inh_synin", "adaptation"]

    def __init__(self, population, file=None):
        super().__init__(population, file=file)
        self.changed_since_last_run = True

    def record(self, variables, ids, sampling_interval=None):
        self.changed_since_last_run = True
        # MADC based recording is only possible for one neuron on a chip
        # it is therefore checked for population size one and no multi
        # assignment before the parent record function is called which modifies
        # state
        if sampling_interval:
            raise NotImplementedError("Sampling interval not implemented.")
        variable_list = pyNN.recording.normalize_variables_arg(variables)
        madc_inter_size = \
            len(set(variable_list).intersection(Recorder.madc_variables))
        if madc_inter_size > 1:
            raise ValueError("Can only set 1 analog record type per chip.")
        if madc_inter_size == 1 and len(ids) != 1:
            raise ValueError("Can only record single neurons via MADC")

        for variable in variable_list:
            if variable in Recorder.madc_variables:
                # get value of the single entry of the set. Only needed for
                # MADC based which is always size 1
                n_id = next(iter(ids))
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
                    raise RuntimeError("Encountered not handled MADC case")

                # check if MADC recorder already set. Ignore if already
                # existing config is set again.
                global_madc_rec = self._simulator.state.madc_recorder
                if global_madc_rec is not None \
                    and (global_madc_rec.cell_id != n_id
                         or global_madc_rec.readout_source != readout_source):
                    raise ValueError(
                        f"Analog record for ID {global_madc_rec.cell_id} of "
                        f"type {global_madc_rec.readout_source} already "
                        "active. Only one concurrent analog readout "
                        "supported.")

                madc_recorder = MADCRecorderSetting(
                    cell_id=n_id, readout_source=readout_source)
                self._simulator.state.madc_recorder = madc_recorder

        super().record(
            variables=variables,
            ids=ids,
            sampling_interval=sampling_interval)

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
        """Returns a dict containing the neuron_id and its spiketimes."""
        all_spiketimes = {}
        for cell_id in ids:
            index_on_pop = np.where(
                self.population.all_cells == int(cell_id))[0]
            assert len(index_on_pop) == 1
            neuron_idx = (simulator.state.populations.index(self.population),
                          index_on_pop[0],
                          halco.CompartmentOnLogicalNeuron().value())
            if neuron_idx in simulator.state.spikes:
                all_spiketimes[cell_id] = simulator.state.spikes[neuron_idx]
        return all_spiketimes

    # TODO: Patch to utilize IrregularlySampledSignal as return value.
    #       Remove when upstream pyNN support is merged and reimplement
    #       _get_all_signals()
    #       https://github.com/NeuralEnsemble/PyNN/pull/754
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
                    times = pq.Quantity(data.get(int(identifier), []), pq.ms)
                    if times.size > 0 and times.max() > t_stop:
                        warn("Recorded at least one spike after t_stop")
                        times = times[times <= t_stop]
                    segment.spiketrains.append(
                        neo.SpikeTrain(
                            times,
                            t_start=self._recording_start_time,
                            t_stop=t_stop,
                            units='ms',
                            source_population=self.population.label,
                            source_id=int(identifier),
                            source_index=self.population.id_to_index(
                                int(identifier)))
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
                if simulator.state.times.size > 0:
                    units = self.population.find_units(variable)
                    source_ids = np.fromiter(ids, dtype=int)
                    id_array = np.array(
                        [self.population.id_to_index(myid) for myid in ids])
                    signal = neo.IrregularlySampledSignal(
                        times=simulator.state.times,
                        signal=simulator.state.madc_samples,
                        units=units,
                        time_units='ms',
                        name=variable,
                        source_population=self.population.label,
                        source_ids=source_ids,
                        array_annotations={"channel_index": id_array})
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
