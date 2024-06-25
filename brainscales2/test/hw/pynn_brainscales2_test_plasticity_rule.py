#!/usr/bin/env python

import textwrap
import unittest
from pynn_brainscales.brainscales2.examples.plasticity_rule import main, \
    cell_params
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v3 import hal, halco


class TestPlasticityRule(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main(cell_params)

    def test_write_read_ppu_symbols(self):
        class PlasticityRule(pynn.PlasticityRule):
            def __init__(self, timer: pynn.Timer):
                pynn.PlasticityRule.__init__(self, timer, observables={})

            def generate_kernel(self) -> str:
                """
                Generate plasticity rule kernel to be compiled into PPU
                program.

                :return: PPU-code of plasticity-rule kernel as string.
                """
                return textwrap.dedent("""
                #include "grenade/vx/ppu/neuron_view_handle.h"
                #include "grenade/vx/ppu/synapse_array_view_handle.h"
                #include <array>
                using namespace grenade::vx::ppu;
                volatile uint32_t test;

                void PLASTICITY_RULE_KERNEL(
                    std::array<SynapseArrayViewHandle, 1>& synapses,
                    std::array<NeuronViewHandle, 0>& /* neurons */)
                {
                    static_cast<void>(test);
                }
                """)

        injected_config = pynn.InjectedConfiguration()
        expectation = hal.PPUMemoryBlock(halco.PPUMemoryBlockSize(1))
        expectation.words = [hal.PPUMemoryWord(
            hal.PPUMemoryWord.Value(0x1234567))]
        injected_config.ppu_symbols = {"test": {
            halco.HemisphereOnDLS.top: expectation,
            halco.HemisphereOnDLS.bottom: expectation}}
        injected_readout = pynn.InjectedReadout()
        injected_readout.ppu_symbols = {"test"}

        pynn.setup(
            injected_config=injected_config,
            injected_readout=injected_readout)

        nrn = pynn.Population(1, pynn.cells.HXNeuron())

        pop_input = pynn.Population(1, pynn.cells.SpikeSourceArray())

        timer = pynn.Timer(start=5, period=10, num_periods=1)
        synapse = pynn.standardmodels.synapses.PlasticSynapse(
            weight=63, plasticity_rule=PlasticityRule(timer=timer))

        pynn.Projection(
            pop_input, nrn, pynn.AllToAllConnector(),
            synapse_type=synapse)

        pynn.run(10)

        actual = pynn.get_post_realtime_read_ppu_symbols()
        self.assertEqual(actual, injected_config.ppu_symbols)

        pynn.end()


if __name__ == "__main__":
    unittest.main()
