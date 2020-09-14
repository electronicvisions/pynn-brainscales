import numpy as np
import pyNN.recording
from pynn_brainscales.brainscales2 import simulator


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator
    state_dt = _simulator.state.dt

    def __init__(self, population, file=None):
        self._simulator.state.dt = self.state_dt
        assert self._simulator.state.dt == 3.4e-05
        super(Recorder, self).__init__(population, file=file)

    def _record(self, variable, new_ids, sampling_interval=None):
        """
        Add the cells in `new_ids` to the set of recorded cells for the
        given variable.
        """

        if sampling_interval:
            raise ValueError("Can't customize sampling interval.")

        assert self.population.celltype.can_record(variable)
        if variable == "v" and len(new_ids) != 1:
            raise ValueError("""Can only record membrane potential of a
                population with size 1.""")

        self.recorded[variable] = new_ids

    # TODO: cf. feature #3598
    def _reset(self):
        raise NotImplementedError

    # TODO: cf. feature #3598
    def _clear_simulator(self):
        raise NotImplementedError

    @staticmethod
    def _get_spiketimes(ids):
        """Returns a dict containing the neuron_id and its spiketimes."""
        all_spiketimes = {}
        if len(simulator.state.spikes) > 0:
            neuron_ids = simulator.state.spikes[:, 0]
            spiketimes = simulator.state.spikes[:, 1]
            for cell_id in ids:
                result_indices = np.where(
                    neuron_ids == simulator.state.neuron_placement[cell_id])
                spikes = spiketimes[result_indices]
                all_spiketimes[cell_id] = spikes
        return all_spiketimes

    # pylint: disable=unused-argument
    @staticmethod
    def _get_all_signals(variable, ids, clear=False):
        if variable == "v":
            signals = np.array(list(zip(simulator.state.times,
                                        simulator.state.membrane)))
        else:
            raise ValueError("Only implemented for membrane potential 'v'")
        return signals

    def _local_count(self, variable, filter_ids):
        counts = {}
        if variable == "spikes":
            for filter_id in self.filter_recorded(variable, filter_ids):
                counts[int(filter_id)] = len(simulator.state.spikes)
        else:
            raise ValueError("Only implemented for spikes")
        return counts
