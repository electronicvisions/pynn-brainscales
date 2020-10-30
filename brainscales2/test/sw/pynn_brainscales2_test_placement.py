#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


class TestPlacement(unittest.TestCase):

    def setUp(self):
        self.permutation = [42, 23, 14]

    def test_mixed_celltypes(self):
        pynn.setup(neuronPermutation=self.permutation)

        pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=[0]))
        pynn.Population(2, pynn.cells.HXNeuron())
        pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=[0]))
        pynn.Population(1, pynn.cells.HXNeuron())

        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(1),
            self.permutation[0])
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(2),
            self.permutation[1])
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(4),
            self.permutation[2])

        pynn.end()

    def test_overflow(self):
        pynn.setup(neuronPermutation=self.permutation)

        pynn.Population(len(self.permutation), pynn.cells.HXNeuron())
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron())
        pynn.end()

    def test_reset(self):
        pynn.setup(neuronPermutation=self.permutation)
        pynn.Population(1, pynn.cells.HXNeuron())
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(0),
            self.permutation[0])
        pynn.reset()
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(0),
            self.permutation[0])
        pynn.end()

    def test_renewed_setup(self):
        pynn.setup(neuronPermutation=self.permutation)
        pynn.Population(1, pynn.cells.HXNeuron())
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(0),
            self.permutation[0])
        pynn.end()
        pynn.setup()
        pynn.Population(1, pynn.cells.HXNeuron())
        self.assertEqual(
            pynn.simulator.state.neuron_placement.id2hwenum(0), 0)
        pynn.end()


if __name__ == '__main__':
    unittest.main()
