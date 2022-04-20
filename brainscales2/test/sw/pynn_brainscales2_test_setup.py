#!/usr/bin/env python

import unittest
from dlens_vx_v3 import halco, lola, sta
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

    def test_unhandled_parameter(self):
        with self.assertRaises(KeyError):
            pynn.setup(foo=True)

    @staticmethod
    def test_injected_config():
        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_non_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
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
            pre_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()},
            post_realtime={halco.AtomicNeuronOnDLS(): lola.AtomicNeuron()}))
        pynn.run(None)
        pynn.end()

    @staticmethod
    def test_injected_builder():
        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_non_realtime=sta.PlaybackProgramBuilder()))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_realtime=sta.PlaybackProgramBuilder()))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            post_realtime=sta.PlaybackProgramBuilder()))
        pynn.run(None)
        pynn.end()

        pynn.setup(injected_config=pynn.InjectedConfiguration(
            pre_non_realtime=sta.PlaybackProgramBuilder(),
            pre_realtime=sta.PlaybackProgramBuilder(),
            post_realtime=sta.PlaybackProgramBuilder()))
        pynn.run(None)
        pynn.end()


if __name__ == '__main__':
    unittest.main()
