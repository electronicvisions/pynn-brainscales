import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn


init_values = {"threshold_v_threshold": 350,
               "leak_v_leak": 1022,
               "leak_i_bias": 320,
               "reset_v_reset": 400,
               "reset_i_bias": 950,
               "reset_enable_multiplication": True,
               "threshold_enable": True,
               "membrane_capacitance_capacitance": 63,
               "refractory_period_refractory_time": 95}


def main(initial_values: dict):
    log = pynn.logger.get("leak_over_threshold")
    pynn.setup()

    pop2 = pynn.Population(2, pynn.cells.HXNeuron(**initial_values))
    pop1 = pynn.Population(1, pynn.cells.HXNeuron(**initial_values))

    pop1.record(["spikes", "v"])
    pop2.record("spikes")
    pynn.run(0.2)

    spikes1 = pop1.get_data("spikes").segments[0]
    spikes2 = pop2.get_data("spikes").segments[0]

    # TODO: Find out how to extract neuron ids.
    for i, spiketrain in enumerate(spikes2.spiketrains):
        log.INFO("Number of Spikes of Neuron {}: ".format(i + 1),
                 len(spiketrain))
        log.INFO("Spiketimes of Neuron {}: ".format(i + 1), spiketrain)

    spiketimes = spikes1.spiketrains[0]
    log.INFO("Number of spikes of single recorded neuron: ", len(spiketimes))
    log.INFO("Spiketimes of recorded neuron: ", spiketimes)

    mem_v = pop1.get_data("v").segments[0]
    times, membrane = zip(*mem_v.filter(name="v")[0])
    log.INFO("Number of MADC Samples: ", len(times))

    plt.figure()
    plt.xlabel("Time [ms]")
    plt.ylabel("Membrane Potential [LSB]")
    plt.plot(times, membrane)
    plt.savefig("plot_tauref95_3.pdf")
    plt.close()

    pynn.end()


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    main(init_values)
