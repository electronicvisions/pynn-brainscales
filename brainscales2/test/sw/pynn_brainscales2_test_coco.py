#!/usr/bin/env python

import unittest
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v2 import halco, lola


class TestCoCoInject(unittest.TestCase):

    def setUp(self):
        coord0 = halco.AtomicNeuronOnDLS(halco.common.Enum(0))
        an0 = lola.AtomicNeuron()
        an0.leak.i_bias = 555
        coord1 = halco.AtomicNeuronOnDLS(halco.common.Enum(1))
        an1 = lola.AtomicNeuron()
        an1.threshold.v_threshold = 123
        self.coco = {coord0: an0, coord1: an1}

    def test_default(self):
        pynn.setup()
        pop = pynn.Population(2, pynn.cells.HXNeuron(self.coco))
        self.assertTrue(np.array_equal(pop.get("leak_i_bias"), [555, 0]))
        self.assertTrue(
            np.array_equal(pop.get("threshold_v_threshold"), [0, 123]))
        pynn.end()

    def test_missing_coco_entries(self):
        pynn.setup()
        pynn.Population(2, pynn.cells.HXNeuron(self.coco))
        with self.assertRaises(KeyError):
            pynn.Population(1, pynn.cells.HXNeuron(self.coco))
        pynn.end()

    def test_permutation(self):
        coord = halco.AtomicNeuronOnDLS(halco.common.Enum(8))
        neuron = lola.AtomicNeuron()
        neuron.leak.i_bias = 666
        pynn.setup(neuronPermutation=[8])
        pop = pynn.Population(1, pynn.cells.HXNeuron({coord: neuron}))
        self.assertEqual(pop.get("leak_i_bias"), 666)
        pynn.end()

    def test_permutation_mismatch(self):
        pynn.setup(neuronPermutation=[8, 14])
        with self.assertRaises(KeyError):
            pynn.Population(2, pynn.cells.HXNeuron(self.coco))
        pynn.end()

    def test_user_overwrite(self):
        pynn.setup()
        pop = pynn.Population(
            2, pynn.cells.HXNeuron(self.coco, leak_i_bias=[100, 150]))
        self.assertTrue(np.array_equal(pop.get("leak_i_bias"), [100, 150]))
        pynn.end()


if __name__ == '__main__':
    unittest.main()
