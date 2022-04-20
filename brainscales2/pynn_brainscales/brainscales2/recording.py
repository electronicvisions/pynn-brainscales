from typing import NamedTuple
import numpy as np
import pyNN.recording
import pyNN.errors
from pynn_brainscales.brainscales2 import simulator
from dlens_vx_v3 import hal


class MADCRecorderSetting(NamedTuple):
    cell_id: int
    readout_source: hal.NeuronConfig.ReadoutSource


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator
    madc_variables = ["v", "exc_synin", "inh_synin", "adaptation"]

    def __init__(self, population, file=None):
        super(Recorder, self).__init__(population, file=file)
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

        super(Recorder, self).record(
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

    # TODO: cf. feature #3598
    def _clear_simulator(self):
        raise NotImplementedError

    @staticmethod
    def _get_spiketimes(ids):
        """Returns a dict containing the neuron_id and its spiketimes."""
        all_spiketimes = {}
        for cell_id in ids:
            neuron_idx = simulator.state.neuron_placement.id2hwenum(
                cell_id)
            if neuron_idx in simulator.state.spikes:
                all_spiketimes[cell_id] = simulator.state.spikes[neuron_idx]
        return all_spiketimes

    # pylint: disable=unused-argument
    @staticmethod
    def _get_all_signals(variable, ids, clear=False):
        if variable in Recorder.madc_variables:
            signals = np.stack(
                (simulator.state.times, simulator.state.madc_samples)).T
        else:
            raise ValueError("Only implemented for membrane potential 'v' and"
                             + "technical parameters: '{exc,inh}_synin', "
                             + "'adaptation'.")
        return signals

    def _local_count(self, variable, filter_ids):
        counts = {}
        if variable == "spikes":
            for filter_id in self.filter_recorded(variable, filter_ids):
                counts[int(filter_id)] = len(simulator.state.spikes)
        else:
            raise ValueError("Only implemented for spikes")
        return counts
