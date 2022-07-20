#!/usr/bin/env python

import textwrap
import numpy as np
import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn


cell_params = {"threshold_v_threshold": 300,
               "leak_v_leak": 750,
               "leak_i_bias": 420,
               "leak_enable_division": True,
               "reset_v_reset": 200,
               "reset_i_bias": 950,
               "reset_enable_multiplication": True,
               "threshold_enable": True,
               "membrane_capacitance_capacitance": 4,
               "refractory_period_refractory_time": 250,
               "excitatory_input_enable": True,
               "excitatory_input_i_bias_tau": 150,
               "excitatory_input_i_bias_gm": 200,
               # FIXME: replace by i_drop_input and i_shift_reference
               # "excitatory_input_v_syn": 700
               }


class PlasticSynapse(
        pynn.PlasticityRule,
        pynn.standardmodels.synapses.StaticSynapse):
    def __init__(self, timer: pynn.Timer, weight: float):
        pynn.PlasticityRule.__init__(self, timer, observables={})
        pynn.standardmodels.synapses.StaticSynapse.__init__(
            self, weight=weight)

    def generate_kernel(self) -> str:
        """
        Generate plasticity rule kernel to be compiled into PPU program.

        :return: PPU-code of plasticity-rule kernel as string.
        """
        return textwrap.dedent("""
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "libnux/vx/location.h"
        using namespace grenade::vx::ppu;
        using namespace libnux::vx;
        void PLASTICITY_RULE_KERNEL(
            std::array<SynapseArrayViewHandle, 1>& synapses,
            std::array<PPUOnDLS, 1> synrams)
        {
            PPUOnDLS location;
            get_location(location);
            if (synrams[0] != location) {
                return;
            }
            SynapseArrayViewHandle::Row zeros;
            for (size_t i = 0; i < 256; ++i) {
                zeros[i] = 0;
            }
            for (size_t i = 0; i < synapses[0].rows.size; ++i) {
                if (synapses[0].rows.test(i)) {
                    synapses[0].set_weights(zeros, i);
                }
            }
        }
        """)


def main(params: dict):
    pynn.setup()

    nrn = pynn.Population(1, pynn.cells.HXNeuron(**params))
    nrn.record(["spikes", "v"])

    spike_times = np.arange(0, 10, 0.01)
    pop_input = pynn.Population(1, pynn.cells.SpikeSourceArray,
                                cellparams={"spike_times": spike_times})

    timer = pynn.Timer(start=5, period=10, num_periods=1)
    synapse = PlasticSynapse(timer=timer, weight=63)

    pynn.Projection(
        pop_input, nrn, pynn.AllToAllConnector(),
        synapse_type=synapse)

    pynn.run(10)

    spiketrain = nrn.get_data("spikes").segments[0].spiketrains[0]
    mem_v = nrn.get_data("v").segments[0].irregularlysampledsignals[0]
    membrane_times = mem_v.times
    membrane_voltage = mem_v.magnitude

    pynn.end()

    # Plot data
    plt.figure()
    plt.xlabel("Time [ms]")
    plt.ylabel("Membrane Potential [LSB]")
    plt.plot(membrane_times, membrane_voltage)
    plt.savefig("plot_plasticity_rule.pdf")
    plt.close()

    return spiketrain


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    log = pynn.logger.get("plasticity_rule")
    spiketimes = main(cell_params)

    log.INFO("Number of spikes of stimulated neuron: ", len(spiketimes))
    log.INFO("Spiketimes of stimulated neuron: ", spiketimes)
