#!/usr/bin/env python

import unittest
from dlens_vx_v2 import halco, lola
import pynn_brainscales.brainscales2 as pynn


class TestpyNNSetup(unittest.TestCase):

    @staticmethod
    def test_default_setup():
        pynn.setup()

    def test_non_unique_permutation(self):
        with self.assertRaises(ValueError):
            pynn.setup(neuronPermutation=[8, 8, 8])

    @staticmethod
    def test_short_neuronpermuation():
        pynn.setup(neuronPermutation=[0, 3, 6])

    @staticmethod
    def test_maxsize_neuronpermuation():
        pynn.setup(neuronPermutation=range(halco.AtomicNeuronOnDLS.size))

    def test_too_large_neuronpermuation(self):
        with self.assertRaises(ValueError):
            pynn.setup(
                neuronPermutation=range(
                    halco.AtomicNeuronOnDLS.size + 1))

    def test_too_large_index(self):
        with self.assertRaises(ValueError):
            pynn.setup(neuronPermutation=[halco.AtomicNeuronOnDLS.size + 1])

    def test_negative_index(self):
        with self.assertRaises(ValueError):
            pynn.setup(neuronPermutation=[-1])

    @staticmethod
    def test_injected_config():
        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_non_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            post_non_realtime={halco.AtomicNeuronOnDLS():
                               lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            post_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_non_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()},
            post_non_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()},
            pre_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()},
            post_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()


if __name__ == '__main__':
    unittest.main()
