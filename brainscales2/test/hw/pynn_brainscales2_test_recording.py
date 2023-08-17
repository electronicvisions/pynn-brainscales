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

        # Once record whole population, once subset and once no neurons
        pops[0].record(['spikes'])
        pynn.PopulationView(pops[1], [0]).record(['spikes'])

        # Spike input
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=np.arange(0.1, runtime, runtime / 10)))
        for pop in pops:
            pynn.Projection(input_pop, pop, pynn.AllToAllConnector(),
                            synapse_type=StaticSynapse(weight=63))

        pynn.run(runtime)

        # Assert correct number of recorded spike trains
        self.assertEqual(len(pops[0].get_data().segments[0].spiketrains),
                         pop_size)
        self.assertEqual(len(pops[1].get_data().segments[0].spiketrains), 1)
        self.assertEqual(len(pops[2].get_data().segments[0].spiketrains), 0)

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


class TestMembraneRecording(unittest.TestCase):
    """
    Tests to ensure the correct recording of analog traces.
    """

    def setUp(self):
        pynn.setup()

    def tearDown(self):
        pynn.end()

    def test_analog_recording(self):
        """
        Here we test that an analog trace is recorded for (and only for) the
        indicated neuron.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop = pynn.Population(2, pynn.cells.HXNeuron())
        pop[0:1].record(["v"])  # slicing to PopulationView sad but necessary

        pynn.run(0.42)

        # check that analog samples were recorded for the target neuron
        samples = pop[0:1].get_data("v").segments[-1]\
            .irregularlysampledsignals[0]
        self.assertTrue(samples.size > 0)

        # check that the recorded samples are assigned only to the selected
        # neuron (and not the other one)
        with self.assertRaises(IndexError):
            samples = pop[1:2].get_data("v").segments[-1]\
                .irregularlysampledsignals[0]

    def test_2ch_analog_recording_split(self):
        """
        Here we test that an analog trace is recorded for (and only for) the
        indicated neurons.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop = pynn.Population(3, pynn.cells.HXNeuron())
        pop[0:1].record(["v"])
        pop[1:2].record(["v"])

        pynn.run(0.42)

        # check that analog samples were recorded for the target neuron
        samples_0 = pop[0:1].get_data("v").segments[-1]\
            .irregularlysampledsignals[0]
        self.assertTrue(samples_0.size > 0)

        samples_1 = pop[1:2].get_data("v").segments[-1]\
            .irregularlysampledsignals[0]
        self.assertTrue(samples_1.size > 0)

        # check that the recorded samples are assigned only to the selected
        # neurons (and not the other one)
        with self.assertRaises(IndexError):
            samples_0 = pop[2:3].get_data("v").segments[-1]\
                .irregularlysampledsignals[0]

    def test_2ch_analog_recording(self):
        """
        Here we test that an analog trace is recorded for (and only for) the
        indicated neurons.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop = pynn.Population(3, pynn.cells.HXNeuron())
        pop[0:2].record(["v"])

        pynn.run(0.42)

        # check that analog samples were recorded for the target neuron
        samples = pop[0:2].get_data("v").segments[-1]\
            .irregularlysampledsignals[0]
        self.assertTrue(samples.size > 0)

        # check that the recorded samples are assigned only to the selected
        # neurons (and not the other one)
        with self.assertRaises(IndexError):
            samples = pop[2:3].get_data("v").segments[-1]\
                .irregularlysampledsignals[0]

    def test_2ch_different_recording(self):
        """
        Test recording of different observables.

        Make sure that different samples are returned for the different
        observables.
        """

        pop = pynn.Population(2, pynn.cells.HXNeuron())
        pop[0:1].record(["v"])
        pop[1:2].record(["adaptation"])

        pynn.run(0.1)

        # check that recorded traces are not identical
        data = pop.get_data().segments[-1].irregularlysampledsignals
        self.assertFalse(np.all(data[0].magnitude == data[1].magnitude))

    def test_reseting(self):
        """
        Test that reseting of recording works correctly.
        """

        pop_a = pynn.Population(1, pynn.cells.HXNeuron())
        pop_b = pynn.Population(1, pynn.cells.HXNeuron())
        pop_c = pynn.Population(1, pynn.cells.HXNeuron())
        pop_a.record(["v"])
        pop_b.record(["v"])

        pynn.run(0.1)

        # check initial config
        self.assertEqual(
            len(pop_a.get_data().segments[-1].irregularlysampledsignals), 1)
        self.assertEqual(
            len(pop_b.get_data().segments[-1].irregularlysampledsignals), 1)
        self.assertEqual(
            len(pop_c.get_data().segments[-1].irregularlysampledsignals), 0)

        pop_a.record(None)
        pop_c.record(["v"])

        # test that samples of `pop_a` are not assigned to `pop_c`
        self.assertEqual(
            len(pop_c.get_data().segments[-1].irregularlysampledsignals), 0)

        pynn.run(0.1)

        self.assertEqual(
            len(pop_a.get_data().segments[-1].irregularlysampledsignals), 0)
        self.assertEqual(
            len(pop_b.get_data().segments[-1].irregularlysampledsignals), 1)
        self.assertEqual(
            len(pop_c.get_data().segments[-1].irregularlysampledsignals), 1)


if __name__ == "__main__":
    unittest.main()
