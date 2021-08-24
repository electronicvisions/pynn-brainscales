import numpy as np
import matplotlib.pyplot as plt

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.morphology import create_mc_neuron, \
    Compartment, SharedLineConnection


def main():
    '''
    Create a chain of compartments with three compartments.

    Record the membrane voltage in the first compartment while synaptic inputs
    are injected into it.
    A plot in which the membrane voltage and recorded spikes are displayed is
    saved to disk.
    '''
    compartments = [
        Compartment(positions=[0], label='my_label',
                    connect_shared_line=[0]),
        Compartment(positions=[1, 2], label='my_label',
                    connect_conductance=[(1, 200)],
                    connect_shared_line=[2]),
        Compartment(positions=[3], label='my_label',
                    connect_conductance=[(3, 200)])]
    connections = [SharedLineConnection(start=0, stop=1, row=0),
                   SharedLineConnection(start=2, stop=3, row=0)]
    McNeuron = create_mc_neuron('McNeuron',
                                compartments=compartments,
                                connections=connections)

    pynn.setup(initial_config=pynn.helper.chip_from_nightly())

    spike_times = np.arange(0.01, 0.2, 0.05)

    pop_input = pynn.Population(5, pynn.cells.SpikeSourceArray,
                                cellparams={"spike_times": spike_times})

    pop = pynn.Population(1, McNeuron())
    pop.record(["spikes", "v"])

    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
    pynn.Projection(pop_input, pop, pynn.AllToAllConnector(),
                    synapse_type=synapse)

    pynn.run(0.2)

    spiketrain = pop.get_data("spikes").segments[0].spiketrains[0]
    mem_v = pop.get_data("v").segments[0].irregularlysampledsignals[0]
    membrane_times = mem_v.times
    membrane_voltage = mem_v.magnitude

    pynn.end()

    # Plot data
    fig, ax = plt.subplots()
    ax.plot(membrane_times, membrane_voltage)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Membrane Potential [LSB]")

    for spike in spiketrain:
        ax.axvline(spike, c='r')

    fig.savefig("multicompartment_example.png")


if __name__ == "__main__":
    main()
