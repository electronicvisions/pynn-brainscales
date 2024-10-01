#!/usr/bin/env python

import numpy as np
import pynn_brainscales.brainscales2 as pynn


def main():
    """
    Experiment with a network which needs to be partitioned.
    Construct and execute a network consisting of an input population
    (size 10), and a hidden as well as an output population (size 1000).
    Since both populations are too large to fit onto a single chip, they are
    splitted and partitioned into multiple executions.
    """
    pynn.setup(enable_neuron_bypass=True)

    input_pop = pynn.Population(
        10, pynn.cells.SpikeSourceArray(spike_times=np.linspace(0., 1., 50)))

    hidden_pop = pynn.Population(1000, pynn.cells.HXNeuron())

    output_pop = pynn.Population(1000, pynn.cells.HXNeuron())
    output_pop.record(["spikes"])

    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)

    pynn.Projection(input_pop, hidden_pop, pynn.AllToAllConnector(),
                    synapse_type=synapse)

    pynn.Projection(hidden_pop, output_pop, pynn.OneToOneConnector(),
                    synapse_type=synapse)

    pynn.run(1)

    spiketrain = output_pop.get_data("spikes").segments[0].spiketrains

    pynn.end()

    return spiketrain


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.DEBUG)
    log = pynn.logger.get("partitioned_network")
    spiketimes = main()

    log.INFO("Number of spikes of output neurons: ",
             sum(len(spikes) for spikes in spiketimes))
