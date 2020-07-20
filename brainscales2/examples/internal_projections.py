from copy import deepcopy
import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v1 import logger


def main():
    main_log = logger.get("internal_projections_main")

    pynn.setup()

    lot_values = {"threshold_v_threshold": 400,
                  "leak_reset_leak_v_leak": 1022,
                  "leak_reset_reset_v_reset": 50,
                  "leak_reset_leak_i_bias": 420,
                  "leak_reset_reset_i_bias": 950,
                  "leak_reset_leak_enable_division": True,
                  "threshold_enable": True,
                  "leak_reset_reset_enable_multiplication": True,
                  "membrane_capacitance_capacitance": 10,
                  "refractory_period_refractory_time": 95}

    init_values = deepcopy(lot_values)
    init_values.update({"leak_reset_leak_v_leak": 650,
                        "leak_reset_reset_v_reset": 650,
                        "excitatory_input_enable": True,
                        "excitatory_input_i_bias_res": 150,
                        "excitatory_input_i_bias_gm": 200,
                        "excitatory_input_v_syn": 700})

    # not leak over threshold
    pop1 = pynn.Population(1, pynn.standardmodels.cells.HXNeuron,
                           initial_values=init_values)

    # leak over threshold
    pop2 = pynn.Population(100, pynn.standardmodels.cells.HXNeuron,
                           initial_values=lot_values)

    pop1.record(["spikes", "v"])
    pop2.record("spikes")

    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
    pynn.Projection(pop2, pop1, pynn.AllToAllConnector(),
                    synapse_type=synapse)

    pynn.run(0.2)

    spiketimes1 = pop1.get_data("spikes").segments[0].spiketrains[0]
    main_log.INFO("Spiketimes of first neuron: ", spiketimes1)

    spikes2 = pop2.get_data("spikes").segments[0]
    spikenumber2 = 0
    for i in range(len(pop2)):
        spikes = len(spikes2.spiketrains[i])
        spikenumber2 += spikes
    main_log.INFO("Number of spikes from pop2: ", spikenumber2)

    mem_v = pop1.get_data("v").segments[0]
    membrane_times, membrane_voltage = zip(*mem_v.filter(name="v")[0])

    pynn.end()

    return len(spiketimes1), membrane_times, membrane_voltage


if __name__ == "__main__":
    log = logger.get("internal_projections")
    spikenumber, times, membrane = main()
    log.INFO("Number of spikes of first neuron: ", spikenumber)

    plt.figure()
    plt.xlabel("Time [ms]")
    plt.ylabel("Membrane Potential [LSB]")
    plt.plot(times, membrane)
    plt.savefig("plot_proj.pdf")
    plt.close()
