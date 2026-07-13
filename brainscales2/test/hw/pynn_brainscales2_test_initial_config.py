#!/usr/bin/env python

import unittest

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse


class TestInitialConfig(unittest.TestCase):
    '''
    Test that initial config is applied and that setting values
    overwrites initial config.`
    '''

    def test_apply(self):
        """
        Test that initial config is applied correctly.

        Load nightly calibration which calibrates the synaptic
        inputs and enables threshold by default.
        Inject many strong spikes and expect that the neuron spikes.
        """
        runtime = 0.1
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())
        pop = pynn.Population(1, pynn.cells.HXNeuron())
        pop.record("spikes")

        source = pynn.Population(10, pynn.cells.SpikeSourceArray(
            spike_times=[runtime / 2]))
        pynn.Projection(source, pop,
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(runtime)

        n_spikes = len(pop.get_data().segments[-1].spiketrains[0])

        self.assertGreater(n_spikes, 0)

        pynn.end()

    def test_overwrite_construction(self):
        """
        Test that overwriting the initial values during construction
        of the neuron works.

        Same experiment as above but this time, we disable the threshold
        during construction and therefore do not expect spikes.
        """
        runtime = 0.1
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())
        pop = pynn.Population(1, pynn.cells.HXNeuron(threshold_enable=False))
        pop.record("spikes")

        source = pynn.Population(10, pynn.cells.SpikeSourceArray(
            spike_times=[runtime / 2]))
        pynn.Projection(source, pop,
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(runtime)

        n_spikes = len(pop.get_data().segments[-1].spiketrains[0])

        self.assertEqual(n_spikes, 0)

        pynn.end()

    def test_overwrite_setting(self):
        """
        Test that overwriting the initial values after construction works.

        Same experiment as above but this time, we disable the threshold
        after construction and therefore do not expect spikes.
        """
        runtime = 0.1
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())
        pop = pynn.Population(1, pynn.cells.HXNeuron())
        pop.record("spikes")

        source = pynn.Population(10, pynn.cells.SpikeSourceArray(
            spike_times=[runtime / 2]))
        pynn.Projection(source, pop,
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(None, command=pynn.RunCommand.PREPARE)
        pop.set(threshold_enable=False)

        pynn.run(runtime)

        n_spikes = len(pop.get_data().segments[-1].spiketrains[0])

        self.assertEqual(n_spikes, 0)

        pynn.end()


if __name__ == "__main__":
    unittest.main()
