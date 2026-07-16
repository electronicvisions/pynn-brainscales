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

    def test_empty_recording(self):
        """
        Test that an empty spike train is returned if no spikes are
        recorded.
        """
        runtime = 0.5  # ms

        pop = pynn.Population(1, pynn.cells.HXNeuron())
        pop.record(['spikes'])
        pynn.run(runtime)
        spiketrains = pop.get_data().segments[0].spiketrains

        self.assertEqual(len(spiketrains), 1)
        self.assertEqual(len(spiketrains[0]), 0)

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

    def test_switch_spike_recording(self):
        """
        Test, if the returned spiketrains are structured in a list with one
        spiketrain per realtime snippet (if recorded) and if the different
        spiketrains with their individual spike rates each are returned in
        the same order, as configured.
        """
        runtime = 10  # ms, runtime of each realtime snippet / config
        pop = pynn.Population(1, pynn.cells.HXNeuron())

        # initial config (spike recording on, 1000 spikes)
        n_spikes = 1000
        pop.record('spikes')

        # Inject spikes
        spikes_1 = np.linspace(0, runtime, n_spikes)
        # also prepare spikes of second snippet right away
        n_spikes = 2000
        spikes_2 = np.linspace(runtime, 2 * runtime, n_spikes)
        input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray(
            spike_times=np.concatenate([spikes_1, spikes_2])))
        pynn.Projection(input_pop, pop, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        pynn.run(runtime, pynn.RunCommand.APPEND)

        # second config (spike recording off, 2000 spikes)
        # (according spikerate already set above)
        pop.record(None)

        # Inject spikes
        pynn.run(runtime, pynn.RunCommand.APPEND)

        # third config (spike recording on, 3000 spikes)
        n_spikes = 3000
        pop.record('spikes')

        # Inject spikes
        spikes_3 = np.linspace(2 * runtime, 3 * runtime, n_spikes)
        input_pop.set(spike_times=np.concatenate([spikes_1, spikes_2, spikes_3,
                                                  spikes_1 + 3 * runtime]))
        pynn.run(runtime, pynn.RunCommand.APPEND)

        # fourth config (spike recording off, 4000 spikes)
        n_spikes = 4000
        pop.record(None)

        # Inject spikes
        spikes_4 = np.linspace(3 * runtime, 4 * runtime, n_spikes)
        input_pop.set(spike_times=spikes_4)
        pynn.run(runtime, pynn.RunCommand.APPEND)

        # fifth config (spike recording off, 5000 spikes)
        n_spikes = 5000

        # Inject spikes
        spikes_5 = np.linspace(4 * runtime, 5 * runtime, n_spikes)
        input_pop.set(spike_times=spikes_5)
        pynn.run(runtime, pynn.RunCommand.APPEND)

        # sixth config (spike recording on, 6000 spikes)
        n_spikes = 6000
        pop.record('spikes')

        # Inject spikes
        spikes_6 = np.linspace(5 * runtime, 6 * runtime, n_spikes)
        input_pop.set(spike_times=spikes_6)

        # execute hardware run
        pynn.run(runtime, pynn.RunCommand.EXECUTE)

        spiketrains = pop.get_data().segments[0].spiketrains

        # Expect spiketrains only for snippets, in which spike recording was
        # enabled
        self.assertEqual(len(spiketrains), 3)
        # Check now the according attributes of the spiketrains to confirm the
        # correct order.
        # Check the length of the spiketrains to match the above set values.
        # Also check, that the set spiketrains of input_pop are cropped
        # accordingly to the realtime snippet bonds
        self.assertLess(0.95 * len(spikes_1), len(spiketrains[0]))
        self.assertLessEqual(len(spiketrains[0]), len(spikes_1) * 1.05)
        self.assertLess(0.95 * len(spikes_3), len(spiketrains[1]))
        self.assertLessEqual(len(spiketrains[1]), len(spikes_3) * 1.05)
        self.assertLess(0.95 * len(spikes_6), len(spiketrains[2]))
        self.assertLessEqual(len(spiketrains[2]), len(spikes_6) * 1.05)


class TestMADCRecording(unittest.TestCase):
    """
    Tests to ensure the correct recording of analog traces via the MADC.
    """

    def setUp(self):
        # Calibration needed since we want to compare recorded traces
        # to expectation, e.g. to test that the correct readout source
        # is selected.
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

    def tearDown(self):
        pynn.end()

    def test_recording(self):
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

    def test_2ch_recording_split(self):
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

    def test_2ch_recording(self):
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
        self.assertFalse(np.array_equal(data[0].magnitude, data[1].magnitude))

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

    def test_switch_recording(self):
        """
        Test whether there are recorded voltage levels for and only for the
        according realtime snippets, where madc recording was enabled
        """

        runtime = 10  # runtime per realtime snippet in ms

        pop = pynn.Population(1, pynn.cells.HXNeuron())

        pop.record('v')
        pynn.run(runtime, pynn.RunCommand.APPEND)
        pynn.run(runtime, pynn.RunCommand.APPEND)

        pop.record(None)
        pynn.run(runtime, pynn.RunCommand.APPEND)

        pop.record('v')

        pynn.run(runtime)

        samples = pop.get_data('v').segments[0].irregularlysampledsignals

        self.assertTrue(len(samples) == 3)
        self.assertTrue(samples[0].size > 0)
        self.assertTrue(samples[1].size > 0)
        self.assertTrue(samples[2].size > 0)

    def test_readout_source(self):
        """
        Test that the selection of the readout source works.
        """
        pop = pynn.Population(1, pynn.cells.HXNeuron())
        in_pop = pynn.Population(5, pynn.cells.SpikeSourceArray(
            spike_times=[0.5, 0.6]))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
        pynn.Projection(in_pop, pop, pynn.AllToAllConnector(),
                        synapse_type=synapse)
        pop.record("v")
        pynn.run(1)
        pynn.reset()

        samples = pop.get_data().segments[-1]\
            .irregularlysampledsignals[0]

        # check that input spikes have an effect.
        self.assertGreater(samples.max() - samples[:100].mean(), 20)

        # test that chainging the readout source works (synaptic inputs
        # pull the synaptic line down)
        pop.record(None)
        pop.record("exc_synin")
        pynn.run(1)
        samples = pop.get_data().segments[-1]\
            .irregularlysampledsignals[0]
        # also assert that the maximum is not too high. Otherwise we
        # could still record the membrane and detect the reset.
        self.assertLess(samples.max() - samples[:100].mean(), 10)
        self.assertGreater(samples[:100].mean() - samples.min(), 20)

    def test_recording_multiple_pops(self):
        """
        Check that we can record two different populations at the same time.
        NOTE: We do not test the correctness of the hardware configuration.
        """
        pops = [pynn.Population(1, pynn.cells.HXNeuron()) for _ in range(2)]

        for pop in pops:
            pop.record(["v"], device='madc')

        pynn.run(0.1)

        for pop in pops:
            samples = pop.get_data("v").segments[-1].irregularlysampledsignals
            self.assertTrue(len(samples) == 1)
            for sample in samples:
                self.assertTrue(sample.size > 0)

        # check that recordings differ
        samples = [pop.get_data().segments[-1].irregularlysampledsignals[0]
                   for pop in pops]
        # MADC might record different number of samples for different channels.
        # If the number of samples is different, we already know that different
        # channels are returned
        if len(samples[0]) == len(samples[1]):
            self.assertFalse(
                np.all(samples[0].magnitude == samples[1].magnitude))


class TestCADCRecording(unittest.TestCase):
    """
    Tests to ensure the correct recording of analog traces via the CADC.
    """

    def setUp(self):
        # calibration needed for CADC
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

    def tearDown(self):
        pynn.end()

    def test_recording(self):
        """
        Here we test that an analog trace is recorded for (and only for) the
        indicated neuron.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop = pynn.Population(4, pynn.cells.HXNeuron())
        pop[0:3].record(["v"], device='cadc')

        pynn.run(0.1)

        # check that analog samples were recorded for the target neuron
        samples = pop[0:3].get_data("v").segments[-1]\
            .irregularlysampledsignals
        self.assertTrue(len(samples) == 3)
        for sample in samples:
            self.assertTrue(sample.size > 0)

        # check that samples are different for different neurons
        self.assertFalse(np.all(samples[0].magnitude == samples[1].magnitude))

        # check that the recorded samples are assigned only to the selected
        # neuron (and not the other one)
        with self.assertRaises(IndexError):
            samples = pop[3:4].get_data("v").segments[-1]\
                .irregularlysampledsignals[0]

    def test_recording_multiple_pops(self):
        """
        Check that we can record two different populations at the same time.
        NOTE: We do not test the correctness of the hardware configuration.
        """
        n_pops = 2
        pops = [pynn.Population(1, pynn.cells.HXNeuron())
                for _ in range(n_pops)]

        for pop in pops:
            pop.record(["v"], device='cadc')

        pynn.run(0.1)

        for pop in pops:
            samples = pop.get_data().segments[-1].irregularlysampledsignals
            self.assertTrue(len(samples) == 1)
            for sample in samples:
                self.assertTrue(sample.size > 0)

        # check that recordings differ
        samples = [pop.get_data().segments[-1].irregularlysampledsignals[0]
                   for pop in pops]
        self.assertFalse(np.all(samples[0].magnitude == samples[1].magnitude))


class TestMADCCADCRecording(unittest.TestCase):
    """
    Tests to ensure the correct recording of analog traces via the CADC
        and the MADC at the same time.
    """

    def setUp(self):
        # calibration needed for CADC
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

    def tearDown(self):
        pynn.end()

    def test_recording(self):
        """
        Here we test that an analog trace is recorded for (and only for) the
        indicated neuron.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop = pynn.Population(5, pynn.cells.HXNeuron())
        cadc_neurons = pop[0:3]
        cadc_neurons.record(["v"], device='cadc')
        madc_neurons = pop[3:4]
        madc_neurons.record(["v"], device='madc')

        pynn.run(0.1)

        # check CADC
        cadc_samples = cadc_neurons.get_data("v").segments[-1]\
            .irregularlysampledsignals
        self.assertTrue(len(cadc_samples) == 3)
        for sample in cadc_samples:
            self.assertTrue(sample.size > 0)

        # check MADC
        madc_samples = madc_neurons.get_data("v").segments[-1]\
            .irregularlysampledsignals
        self.assertTrue(len(madc_samples) == 1)
        self.assertTrue(madc_samples[0].size > 0)

        # MADC has higher temporal resolution -> make sure correct device is
        # assigned to correct neurons
        self.assertGreater(len(madc_samples[0]), len(cadc_samples[0]))

        # check that the recorded samples are assigned only to the selected
        # neuron (and not the other one)
        with self.assertRaises(IndexError):
            _ = pop[4:5].get_data("v").segments[-1]\
                .irregularlysampledsignals[0]

    def test_recording_multiple_pops(self):
        """
        Check that we can record two different populations at the same time.
        NOTE: We do not test the correctness of the hardware configuration.
        """

        pop_cadc = pynn.Population(3, pynn.cells.HXNeuron())
        pop_cadc.record(["v"], device='cadc')
        pop_madc = pynn.Population(1, pynn.cells.HXNeuron())
        pop_madc.record(["v"], device='madc')

        pynn.run(0.1)

        # check CADC
        cadc_samples = pop_cadc.get_data("v").segments[-1]\
            .irregularlysampledsignals
        self.assertTrue(len(cadc_samples) == 3)
        for sample in cadc_samples:
            self.assertTrue(sample.size > 0)

        # check MADC
        madc_samples = pop_madc.get_data("v").segments[-1]\
            .irregularlysampledsignals
        self.assertTrue(len(madc_samples) == 1)
        self.assertTrue(madc_samples[0].size > 0)

        # MADC has higher temporal resolution -> make sure correct device is
        # assigned to correct neurons
        self.assertGreater(len(madc_samples[0]), len(cadc_samples[0]))


class TestClearBehavior(unittest.TestCase):
    """
    Test clear behavior.

    Test proper behavior of the clear parameter of get_data.
    """
    n_spikes = 10
    runtime = 10  # ms

    def setUp(self) -> None:
        '''
        Perform experiment and record spikes
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
        """
        spikes_0 = self.pop_0.get_data(clear=True).segments[0].spiketrains[0]
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spikes_0), self.n_spikes)
        self.assertEqual(len(spikes_1), self.n_spikes)

        spiketrains_0 = self.pop_0.get_data().segments[0].spiketrains
        spikes_1 = self.pop_1.get_data().segments[0].spiketrains[0]
        self.assertEqual(len(spiketrains_0), 0)
        self.assertEqual(len(spikes_1), self.n_spikes)


class TestMultipleRuns(unittest.TestCase):
    """
    Test recording when several runs are executed.
    """
    runtime = 2

    def setUp(self) -> None:
        '''
        Perform experiment and record spikes
        '''
        pynn.setup(enable_neuron_bypass=True)

        self.pop = pynn.Population(1, pynn.cells.HXNeuron())
        self.pop.record('spikes')

        self.input_pop = pynn.Population(1, pynn.cells.SpikeSourceArray())
        pynn.Projection(self.input_pop, self.pop, pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))
        self.input_pop.record("spikes")

    def tearDown(self) -> None:
        pynn.end()

    def test_raise_without_reset(self):
        pynn.run(self.runtime)

        with self.assertRaises(RuntimeError):
            pynn.run(self.runtime)

    def test_changes(self):
        n_runs = 4

        for n_run in range(n_runs):
            spikes = np.linspace(0.1, 0.9 * self.runtime, n_run)
            self.input_pop.set(spike_times=spikes)
            pynn.run(self.runtime)
            pynn.reset()

        self.assertEqual(len(self.pop.get_data().segments), n_runs)
        self.assertEqual(len(self.input_pop.get_data().segments), n_runs)

        for n_run, seg in enumerate(self.pop.get_data().segments):
            self.assertEqual(len(seg.spiketrains[0]), n_run)

        for n_run, seg in enumerate(self.input_pop.get_data().segments):
            self.assertEqual(len(seg.spiketrains[0]), n_run)

    def test_clear(self):
        n_runs = 4

        for n_run in range(n_runs):
            spikes = np.linspace(0.1, 0.9 * self.runtime, n_run)
            self.input_pop.set(spike_times=spikes)

            pynn.run(self.runtime)
            pynn.reset()
            segments = self.pop.get_data(clear=True).segments
            segments_in = self.input_pop.get_data(clear=True).segments

            self.assertEqual(len(segments), 1)
            self.assertEqual(len(segments_in), 1)

            self.assertEqual(len(segments[-1].spiketrains[0]), n_run)
            self.assertEqual(len(segments_in[-1].spiketrains[0]), n_run)

    def test_with_repetitions(self):
        n_runs = 4
        n_reps = 2

        for n_run in range(n_runs):
            spikes = np.linspace(0.1, 0.9 * self.runtime, n_run)
            self.input_pop.set(spike_times=spikes)

            for _ in range(n_reps):
                pynn.run(self.runtime)
                pynn.reset()
            segments = self.pop.get_data(clear=True).segments
            segments_in = self.input_pop.get_data(clear=True).segments

            self.assertEqual(len(segments), n_reps)
            self.assertEqual(len(segments_in), n_reps)

            for seg in segments:
                self.assertEqual(len(seg.spiketrains[0]), n_run)

            for seg in segments_in:
                self.assertEqual(len(seg.spiketrains[0]), n_run)


if __name__ == "__main__":
    unittest.main()
