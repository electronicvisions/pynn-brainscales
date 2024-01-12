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

    def test_reconfigure_spike_recording(self):
        """
        Test, if the returned spiketrains are structured in a list with one
        spiketrain per realtime snippet and if the different spiketrains
        with their individual spike rates each are returned in the same order,
        as configured.
        """
        runtime = 10  # ms, runtime of each realtime snippet / config
        pop = pynn.Population(1, pynn.cells.HXNeuron())

        # initial config (spike recording on, 1000 spikes)
        n_spikes = 1000
        pop.record('spikes')

        # Inject spikes
        spikes_1 = np.linspace(0, runtime, n_spikes)
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=spikes_1))
        pynn.Projection(input_pop, pop, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        pynn.add(runtime)

        # second config (spike recording off, 2000 spikes)
        n_spikes = 2000
        pop.record(None)

        # Inject spikes
        spikes_2 = np.linspace(0, runtime, n_spikes)
        input_pop.set(spike_times=spikes_2)
        pynn.add(runtime)

        # third config (spike recording on, 3000 spikes)
        n_spikes = 3000
        pop.record('spikes')

        # Inject spikes
        spikes_3 = np.linspace(0, runtime, n_spikes)
        input_pop.set(spike_times=spikes_3)
        pynn.add(runtime)

        # fourth config (spike recording off, 4000 spikes)
        n_spikes = 4000
        pop.record(None)

        # Inject spikes
        spikes_4 = np.linspace(0, runtime, n_spikes)
        input_pop.set(spike_times=spikes_4)
        pynn.add(runtime)

        # fifth config (spike recording off, 5000 spikes)
        n_spikes = 5000

        # Inject spikes
        spikes_5 = np.linspace(0, runtime, n_spikes)
        input_pop.set(spike_times=spikes_5)
        pynn.add(runtime)

        # sixth config (spike recording on, 6000 spikes)
        n_spikes = 6000
        pop.record('spikes')

        # Inject spikes
        spikes_6 = np.linspace(0, runtime, n_spikes)
        input_pop.set(spike_times=spikes_6)

        #execute hardware run
        pynn.run(runtime)

        spiketrains = pop.get_data().segments[0].spiketrains

        # Check now the according attributes of the spiketrains to confirm the
        # correct order.
        # Check the length of the spiketrains to match the above set values,
        # if recorded, but 0 otherwise
        self.assertLess(0.95*len(spikes_1), len(spiketrains[0]))
        self.assertLessEqual(len(spiketrains[0]), len(spikes_1))
        self.assertEqual(len(spiketrains[1]), 0)
        self.assertLess(0.95*len(spikes_3), len(spiketrains[2]))
        self.assertLessEqual(len(spiketrains[2]), len(spikes_3))
        self.assertEqual(len(spiketrains[3]), 0)
        self.assertEqual(len(spiketrains[4]), 0)
        self.assertLess(0.95*len(spikes_6), len(spiketrains[5]))
        self.assertLessEqual(len(spiketrains[5]), len(spikes_6))


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

    def test_resetting(self):
        """
        Test that resetting of recording works correctly.
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

        pynn.reset()

        pop_a.record(None)
        pop_c.record(["v"])

        # test that samples of `pop_a` are not assigned to `pop_c`
        self.assertEqual(
            len(pop_c.get_data().segments[-1].irregularlysampledsignals), 0)
        pynn.reset()

        pynn.run(0.1)

        self.assertEqual(
            len(pop_a.get_data().segments[-1].irregularlysampledsignals), 0)
        self.assertEqual(
            len(pop_b.get_data().segments[-1].irregularlysampledsignals), 1)
        self.assertEqual(
            len(pop_c.get_data().segments[-1].irregularlysampledsignals), 1)

    def test_reconfigure_analog_recording(self):
        """
        Test whether there are recorded voltage levels for and only for the
        according realtime snippets, where madc recording was enabled
        """

        runtime = 10 #runtime per realtime snippet in ms

        pop = pynn.Population(1, pynn.cells.HXNeuron())

        pop.record('v')
        pynn.add(runtime)
        pynn.add(runtime)

        pop.record(None)
        pynn.add(runtime)

        pop.record('v')

        pynn.run(runtime)

        samples = pop.get_data('v').segments[0].irregularlysampledsignals

        self.assertTrue(len(samples) == 3)
        self.assertTrue(samples[0].size > 0)
        self.assertTrue(samples[1].size > 0)
        self.assertTrue(samples[2].size > 0)


class TestClearBehaviour(unittest.TestCase):
    """
    Test clear behaviour.

    Test proper behaviour of the clear parameter of get_data.
    """
    n_spikes = 10
    runtime = 10  # ms

    def setUp(self) -> None:
        '''
        Perform experiment an record spikes
        '''
        pynn.setup(enable_neuron_bypass=True)

        pop_0 = pynn.Population(1, pynn.cells.HXNeuron())
        pop_1 = pynn.Population(1, pynn.cells.HXNeuron())
        pop_0.record('spikes')
        pop_1.record('spikes')

        # Inject spikes
        spikes = np.linspace(0.1, 0.9 * self.runtime, self.n_spikes)
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=spikes))
        pynn.Projection(input_pop, pop_0, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        pynn.Projection(input_pop, pop_1, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        pynn.run(self.runtime)

        self.pop_0 = pop_0
        self.pop_1 = pop_1

    def tearDown(self) -> None:
        pynn.end()

    def test_no_clear(self):
        """
        Test without clear.

        All data should still be available.
        """

        spikes_0 = self.pop_0.get_data().segments[0].spiketrains[0]
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spikes_0), self.n_spikes)
        self.assertEqual(len(spikes_1), self.n_spikes)

        spikes_0 = self.pop_0.get_data().segments[0].spiketrains[0]
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spikes_0), self.n_spikes)
        self.assertEqual(len(spikes_1), self.n_spikes)

    def test_only_clear_one_population(self):
        """
        Only clear recording from one population.

        This should not affect the data of the other population.
        be available.
        """
        spikes_0 = self.pop_0.get_data(clear=True).segments[0].spiketrains[0]
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spikes_0), self.n_spikes)
        self.assertEqual(len(spikes_1), self.n_spikes)

        spikes_0 = self.pop_0.get_data().segments[0].spiketrains[0]
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spikes_0), 0)
        self.assertEqual(len(spikes_1), self.n_spikes)


if __name__ == "__main__":
    unittest.main()
