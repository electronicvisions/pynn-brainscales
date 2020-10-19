import unittest
import numpy as np
from pyNN.random import RandomDistribution, NumpyRNG
import pynn_brainscales.brainscales2 as pynn


class TestNeuronBypassMode(unittest.TestCase):
    """
    Tests the neuron bypass mode by sending in a Poisson spike train via a
    SpikeSourceArray.
    This spike source is connected to a HXNeuron in bypass mode.
    The input spike train and the output of the HXNeuron are compared.

    :cvar rate: Rate of the Poisson spike train in Hz.
    :cvar runtime: Runtime of experiment in ms.
    """

    runtime = 10000  # ms
    rate = 200  # Hz

    @staticmethod
    def poisson_spike_train(rate: float, start: float = 0,
                            stop: float = runtime,
                            seed: int = 4245) -> np.array:
        """
        Generates a Poisson spike train.

        :param rate: Rate of the train in Hz.
        :param stop: Stop time of the spike train in ms.
        :param start: Start time of the spike train in ms.
        :param seed: Seed to use for the random number generator.

        :return: Spike times of Poisson spike train as an array.
        """
        assert start < stop, "Start time has to be shorter than stop time"

        # Use period in us to support non-integer spike times in ms
        period = 1 / rate * 1e6  # period in us

        poisson_dist = RandomDistribution("poisson", lambda_=period,
                                          rng=NumpyRNG(seed=seed))

        # Generate spike times till the stop time is exceeded
        spikes = []
        time = start
        while True:
            time += poisson_dist.next() / 1000  # convert from us to ms
            if time > stop:
                break
            spikes.append(time)

        return np.array(sorted(spikes))

    def setUp(self):
        pynn.setup(enable_neuron_bypass=True)

    def tearDown(self):
        pynn.end()

    def test_bypass(self):
        # Create network with single SpikeSourceArray which projects on a
        # single HX neuron
        spikes_in = self.poisson_spike_train(self.rate)
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray,
                                    cellparams={"spike_times": spikes_in})

        output_pop = pynn.Population(1, pynn.cells.HXNeuron)
        output_pop.record(["spikes"])

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
        pynn.Projection(input_pop, output_pop, pynn.AllToAllConnector(),
                        synapse_type=synapse)

        # Run experiment on chip
        pynn.run(self.runtime)

        # Make sure that at least 98% (estimate considering typical spike
        # loss) of the spikes are read back and that the time difference
        # between them is less than 0.01ms
        spikes_out = output_pop.get_data().segments[0].spiketrains[0].magnitude

        # Add a small number such that timestamps with a '5' the third decimal
        # place are rounded up to the next higher (and not to the next even)
        # number
        input_set = set(np.round(spikes_in + 1e-9, 2))
        output_set = set(np.round(spikes_out + 1e-9, 2))
        assert len(input_set - output_set) / len(input_set) < 0.02


if __name__ == "__main__":
    unittest.main()
