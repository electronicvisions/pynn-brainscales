#!/usr/bin/env python

import unittest
import numpy as np

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse


class TestSpikeRecording(unittest.TestCase):
    """
    Tests to ensure the recording of spikes works correctly.
    """

    def setUp(self):
        pynn.setup(enable_neuron_bypass=True)

    def tearDown(self):
        pynn.end()

    def test_number_of_spike_trains(self):
        """
        Here we test that only spikes are recorded for the neurons we set in
        the spike recording mode.
        """

        runtime = 0.5  # ms
        pop_size = 4

        pops = []
        pops.append(pynn.Population(pop_size, pynn.cells.HXNeuron()))
        pops.append(pynn.Population(pop_size, pynn.cells.HXNeuron()))
        pops.append(pynn.Population(pop_size, pynn.cells.HXNeuron()))
        all_neurons = pynn.Assembly(*pops)

        # Once record whole population, once subset and once no neurons
        pops[0].record(['spikes'])
        pynn.PopulationView(pops[1], [0]).record(['spikes'])

        # Spike input
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=np.arange(0.1, runtime, runtime / 10)))
        pynn.Projection(input_pop, all_neurons, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(runtime)

        # Assert correct number of recorded spike trains
        self.assertEqual(len(pops[0].get_data().segments[0].spiketrains),
                         pop_size)
        self.assertEqual(len(pops[1].get_data().segments[0].spiketrains), 1)
        self.assertEqual(len(pops[2].get_data().segments[0].spiketrains), 0)

        self.assertEqual(len(all_neurons.get_data().segments[0].spiketrains),
                         pop_size + 1)

    def test_timing_of_spikes(self):
        """
        Test whether the output spikes have approximately the same timing as
        the input spikes.
        """

        runtime = 100  # ms
        n_spikes = 1000

        pop_1 = pynn.Population(1, pynn.cells.HXNeuron())
        pop_1.record('spikes')
        pop_2 = pynn.Population(1, pynn.cells.HXNeuron())
        pop_2.record('spikes')

        # Inject spikes
        spikes_1 = np.arange(0.1, runtime, runtime / n_spikes)
        input_pop_1 = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=spikes_1))
        pynn.Projection(input_pop_1, pop_1, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        # Take midpoints of spikes_1 to have some variation in timing and
        spikes_2 = (spikes_1[:-1] + spikes_1[1:]) / 2
        input_pop_2 = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=spikes_2))
        pynn.Projection(input_pop_2, pop_2, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(runtime)

        # Make sure that at least 98% (estimate considering typical spike
        # loss) of the spikes are read back and that the time difference
        # between them is less than 0.01ms.
        # To accomplish that we round the spike times to two decimal places
        # and then compare the sets of the input and output spike times.
        # By default numpy does not round '5' to the next higher number but
        # to the next even number, e.g. np.round(2.5) == 2. Therefore, we
        # add a small number such that timestamps with a '5' at the third
        # decimal place are rounded up to the next higher (and not even) number
        input_set_1 = set(np.round(spikes_1 + 1e-9, 2))
        output_set_1 = set(np.round(
            pop_1.get_data().segments[0].spiketrains[0].magnitude + 1e-9, 2))
        self.assertLess(len(input_set_1 - output_set_1) / len(input_set_1),
                        0.02)

        input_set_2 = set(np.round(spikes_2 + 1e-9, 2))
        output_set_2 = set(np.round(
            pop_2.get_data().segments[0].spiketrains[0].magnitude + 1e-9, 2))
        self.assertLess(len(input_set_2 - output_set_2) / len(input_set_2),
                        0.02)


if __name__ == "__main__":
    unittest.main()
