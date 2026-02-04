#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.morphology import create_mc_neuron, \
    Compartment, SharedLineConnection


def create_neuron_class(length: int):
    """
    Create a neuron class for a dendritic chain.

    :param length: Number of compartments in the chain.
    """
    #
    #    ┌─┐ ┌─┐       ┌─┐
    #    $ │ $ │       $ │
    #  0─0 1─1 2─...─N-1 N─N
    compartments = []
    connections = []
    for n_comp in range(length):
        first = n_comp * 2
        second = first + 1

        connect_shared_line = []
        if n_comp > 0:
            connect_shared_line.append(first)

        connect_conductance = []
        if n_comp < length - 1:
            # connect to the right
            connect_conductance.append((second, 200))
            connections.append(
                SharedLineConnection(start=second,
                                     stop=second + 1,
                                     row=0))
        compartments.append(
            Compartment(positions=[first, second],
                        label=f'comp{n_comp}',
                        connect_shared_line=connect_shared_line,
                        connect_conductance=connect_conductance))

    return create_mc_neuron('McNeuron',
                            compartments=compartments,
                            connections=connections,
                            single_active_circuit=True)


def main(length: int = 3,
         runtime: float = 0.15):
    '''
    Create a chain of compartments.

    Record the membrane voltage in each compartment while inputs are injected
    in the different compartments.

    :param length: Length of the chain.
    :param runtime: Experiment runtime in ms. Note that for runtimes over
        around 0.15 ms, samples from the CADC are lost, i.e. the data is
        not recorded of the whole experiment time.
    '''
    McNeuron = create_neuron_class(length)

    pynn.setup(initial_config=pynn.helper.chip_from_nightly())

    pop = pynn.Population(1, McNeuron(threshold_enable=False))

    # Distribute spikes over the experiment runtime
    spike_times = np.linspace(0.01, runtime - 0.05, length)

    # Inject one spike in each compartment
    for n_input in range(length):
        in_pop = pynn.Population(3, pynn.cells.SpikeSourceArray(
            spike_times=[spike_times[n_input]]))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
        pynn.Projection(
            in_pop, pop,
            pynn.AllToAllConnector(location_selector=f'comp{n_input}'),
            synapse_type=synapse)

    # record voltage in each compartment
    pop.record(
        "v", locations=[f"comp{n_comp}" for n_comp in range(length)],
        device='cadc')

    pynn.run(runtime)

    v_mems = pop.get_data("v").segments[0].irregularlysampledsignals

    pynn.end()

    # Plot data
    fig, axs = plt.subplots(length, sharex=True, sharey=True)

    for n_comp, (ax, v_mem) in enumerate(zip(axs, v_mems)):
        ax.set_title(f'Compartment {n_comp}')
        ax.plot(v_mem.times, v_mem)
        ax.axvline(spike_times[n_comp], c='b', ls=':')
    axs[int(length / 2)].set_ylabel("Membrane Potential [LSB]")
    axs[-1].set_xlabel("Time [ms]")

    fig.savefig("multicompartment_example.png")


if __name__ == "__main__":
    main()
