import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v1 import logger


init_values = {"threshold_v_threshold": 300,
               "leak_reset_leak_v_leak": 750,
               "leak_reset_reset_v_reset": 200,
               "leak_reset_leak_i_bias": 420,
               "leak_reset_reset_i_bias": 950,
               "leak_reset_leak_enable_division": True,
               "threshold_enable": True,
               "leak_reset_reset_enable_multiplication": True,
               "membrane_capacitance_capacitance": 4,
               "refractory_period_refractory_time": 250,
               "excitatory_input_enable": True,
               "excitatory_input_i_bias_res": 150,
               "excitatory_input_i_bias_gm": 200,
               "excitatory_input_v_syn": 700}


def main(initial_values: dict):
    pynn.setup()

    nrn = pynn.Population(1, pynn.cells.HXNeuron,
                          initial_values=initial_values)
    nrn.record(["spikes", "v"])

    spike_times = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
    pop_input = pynn.Population(20, pynn.cells.SpikeSourceArray,
                                cellparams={"spike_times": spike_times})

    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
    pynn.Projection(pop_input, nrn, pynn.AllToAllConnector(),
                    synapse_type=synapse)

    pynn.run(0.2)

    spiketrain = nrn.get_data("spikes").segments[0].spiketrains[0]
    mem_v = nrn.get_data("v").segments[0]
    membrane_times, membrane_voltage = zip(*mem_v.filter(name="v")[0])

    pynn.end()

    return spiketrain, membrane_times, membrane_voltage


if __name__ == "__main__":
    log = logger.get("external_input")
    spiketimes, times, membrane = main(init_values)

    log.INFO("Number of spikes of stimulated neuron: ", len(spiketimes))
    log.INFO("Spiketimes of stimulated neuron: ", spiketimes)

    plt.figure()
    plt.xlabel("Time [ms]")
    plt.ylabel("Membrane Potential [LSB]")
    plt.plot(times, membrane)
    plt.savefig("plot_external_input.pdf")
    plt.close()