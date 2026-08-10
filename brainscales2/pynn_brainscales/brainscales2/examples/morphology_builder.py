#!/usr/bin/env python

"""
We use the morphology builder to define an abstract graph of a
multi-compartmental neuron model and then map it automatically to
BrainScaleS-2.

With a CompartmentBuilder, we can construct different Compartments with
different mechanisms. These compartments can then be connected with the
MorphologyBuilder to define a multi-compartmental neuron model.

We will also show how to record the different mechanisms and how
to connect other neurons to multi-compartmental neurons.

If you want to manually define the mapping from compartments to neuron
circuits, have a look at the example 'manual_morphology_builder.py'.
"""

import numpy as np
import matplotlib.pyplot as plt

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.morphology import CompartmentBuilder, \
    MorphologyBuilder, Connection
from pynn_brainscales.brainscales2.morphology import mechanisms


def create_neuron_class(length: int):
    """
    Create a neuron class for a dendritic chain.

    :param length: Number of compartments in the chain.
    """
    # Construct Compartment Class:
    # The CompartmentBuilder allows to define compartments with different
    # mechanisms. Each mechanism has a label which is later used to
    # address it in order to change parameters, record it or in case of
    # synapses connect to it.
    # Similar to CalibHXNeuronCoba, parameters are calibration targets of
    # calix. Upon experiment execution, a calibration is automatically
    # executed and the corresponding result applied to our neurons.
    comp_builder = CompartmentBuilder()
    comp_builder.add(
        mechanisms.MembraneCapacitance(capacitance=2.2e-12), label='cap')
    comp_builder.add(
        mechanisms.CurrentBasedSynapse(strength=500, time_constant=10e-6),
        label='syn')
    comp_builder.add(
        mechanisms.Leak(v_leak=60, tau_mem=10e-6), label='leak')
    compartment_class = comp_builder.done("Compartment")

    # Construct Neuron Class:
    # We can use the MorphologyBuilder to construct a neuron. We add different
    # compartments to the builder and connect them. Again, each compartment
    # gets a label such that we can later address the different compartments.

    builder = MorphologyBuilder()
    nodes = []
    for n_comp in range(length):
        nodes.append(
            builder.add_compartment(compartment_class(), label=f"C{n_comp}"))

    connections = [Connection(first, second, 10e-6) for first, second in
                   zip(nodes[:-1], nodes[1:])]

    builder.connect(connections)
    return builder.done("MyNeuron")


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
    neuron_class = create_neuron_class(length)

    pynn.setup()

    pop = pynn.Population(1, neuron_class())

    # Distribute spikes over the experiment runtime
    spike_times = np.linspace(0.01, runtime - 0.05, length)

    # Inject one spike in each compartment
    for n_input in range(length):
        in_pop = pynn.Population(3, pynn.cells.SpikeSourceArray(
            spike_times=[spike_times[n_input]]))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
        # When connecting other neurons to our neuron, we can define
        # the compartment with the 'location_selector' and the target
        # receptor with 'receptor_type'.
        pynn.Projection(
            in_pop, pop,
            pynn.AllToAllConnector(location_selector=f'C{n_input}'),
            synapse_type=synapse,
            receptor_type="syn")

    # Some mechanisms such as the capacitance can be recorded.
    # We use their label to record them. The compartments which should be
    # recorded can be specified with the 'locations' argument.
    pop.record(
        "cap", locations=[f"C{n_comp}" for n_comp in range(length)],
        device='cadc')

    # Upon execution, it is checked if the desired parameters have
    # already been calibrated before and if so, the calibration is
    # load from cache. Otherwise, the calibration is run.
    pynn.run(runtime)

    v_mems = pop.get_data().segments[0].irregularlysampledsignals

    pynn.end()

    # Plot data
    fig, axs = plt.subplots(length, sharex=True, sharey=True)

    for n_comp, (ax, v_mem) in enumerate(zip(axs, v_mems)):
        ax.set_title(f'Compartment {n_comp}')
        ax.plot(v_mem.times, v_mem)
        ax.axvline(spike_times[n_comp], c='b', ls=':')
    axs[int(length / 2)].set_ylabel("Membrane Potential [LSB]")
    axs[-1].set_xlabel("Time [ms]")

    fig.savefig("morphology_builder.png")


if __name__ == "__main__":
    main()
