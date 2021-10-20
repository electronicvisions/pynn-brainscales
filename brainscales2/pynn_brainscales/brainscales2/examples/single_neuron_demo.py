import matplotlib.pyplot as plt
import numpy as np

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2 import Population
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

# Welcome to this tutorial of using pyNN for the BrainScaleS-2 neuromorphic
# accelerator.
# We will guide you through all the steps necessary to interact with the
# system and help you explore the capabilities of the on-chip analog neurons
# and synapses.

# To begin with, we configure the logger used during our experiments.
pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
logger = pynn.logger.get("single_neuron_demo")

# -- First experiment: A silent neuron -- #
# As a first experiment, we will record the membrane of a single, silent
# neuron on the analog substrate.
plt.figure()
plt.suptitle("First experiment: A silent neuron")

# The pyNN-interface can be used similarly to existing simulators and other
# neuromorphic platforms.
pynn.setup()

# In the current state, we only expose the neuron type 'HXNeuron',
# which allows low-level access to all circuit parameters. It can be
# configured by passing initial values to the respective Population.
# Each Population may consist of multiple neurons (in this case: one),
# all sharing the same parameters.
# Circuit parameters control the dynamic behavior of the neuron as well as
# static configuration. Most of them are either boolean or given in units of
# 'LSB' for chip-internal Digital-to-Analog converters - they have no direct
# biological translation.
# For this first example, you may alter the leak potential and observe
# the response of the analog neuron's resting potential.
pop = pynn.Population(1, pynn.cells.HXNeuron(
    # Leak potential, range: 300-1000
    leak_v_leak=700,
    # Leak conductance, range: 0-1022
    leak_i_bias=1022))

# The chip contains a fast Analog-to-Digital converter. It can be used to
# record different observables of a single analog neuron - most importantly
# the membrane potential.
#
# The chip additionally includes slower, parallel ADCs which will allow for
# parallel access to analog signals in multiple neurons. Support for this
# ADC will be integrated in future versions of our pyNN-Api.
pop.record(["v"])

# Calling pynn.run(time_in_ms) will as a first step apply the static
# configuration to the neuromorphic substrate. As a second step, the network
# is evolved for a given amount of time and neurons are stimulated by any
# stimuli specified by the user.
# The time is given in units of milliseconds (wall clock time),
# representing the hardware's intrinsic 1000-fold speed-up compared to
# biological systems.
pynn.run(0.2)


# The following helper function plots the membrane potential as well as any
# spikes found during the run. It will be used throughout this example.
def plot_membrane_dynamics(population: Population, segment_id=-1):
    """
    Plot the membrane potential of the neuron in a given population view. Only
    population views of size 1 are supported.
    :param population: Population, membrane traces and spikes are plotted for.
    :param segment_id: Index of the neo segment to be plotted. Defaults to
                       -1, encoding the last recorded segment.
    """
    if len(population) != 1:
        raise ValueError("Plotting is supported for populations of size 1.")

    # Experimental results are given in the 'neo' data format, the following
    # lines extract membrane traces as well as spikes and construct a simple
    # figure.
    mem_v = population.get_data("v").segments[segment_id].analogsignals[0].base
    times = mem_v[:, 0]
    membrane = mem_v[:, 1]
    try:
        spikes = population.get_data("spikes").segments[0]

        for spiketime in spikes.spiketrains[0]:
            plt.axvline(spiketime, color="black")
    except IndexError:
        logger.INFO("No spikes found to plot.")

    plt.plot(times, membrane, alpha=0.5)
    logger.INFO(f"Mean membrane potential: {np.mean(membrane)}")
    plt.xlabel("Wall clock time [ms]")
    plt.ylabel("ADC readout [a.u.]")
    plt.ylim(0, 1023)  # ADC precision: 10bit -> value range: 0-1023


plot_membrane_dynamics(pop)
plt.show()

# Reset the pyNN internal state and prepare for the following experiment.
pynn.end()

# -- Second experiment: Leak over threshold -- #
# As a second experiment, we will let the neurons on BrainScaleS-2 spike by
# setting a 'leak-over-threshold' configuration.
plt.figure()
plt.suptitle("Second experiment: Leak over threshold")
pynn.setup()

# Since spiking behavior requires the configuration of additional circuits
# in the neuron, the initial values for our leak-over-threshold population
# are more complex.
# The different potentials (leak, reset, threshold) have no direct
# correspondence: A configured leak potential of 300 might equal a
# configured threshold potential of value 600 in natural units on the physical
# system.
pop = pynn.Population(1, pynn.cells.HXNeuron(
    # Leak potential, range: 300-1000
    leak_v_leak=1000,
    # Leak conductance, range: 0-1022
    leak_i_bias=200,
    # Threshold potential, range: 0-600
    threshold_v_threshold=300,
    # Reset potential, range: 300-1000
    reset_v_reset=400,
    # Membrane capacitance, range: 0-63
    membrane_capacitance_capacitance=63,
    # Refractory time, range: 0-255
    refractory_period_refractory_time=120,
    # Enable reset on threshold crossing
    threshold_enable=True,
    # Reset conductance, range: 0-1022
    reset_i_bias=1022,
    # Enable strengthening of reset conductance
    reset_enable_multiplication=True))
pop.record(["v", "spikes"])
pynn.run(0.2)
plot_membrane_dynamics(pop)
plt.show()
pynn.end()

# -- Third experiment: Fixed-pattern variations -- #
# Due to the analog nature of the BrainScaleS-2 platform, the inevitable
# mismatch of semiconductor fabrication results in inhomogeneous properties
# of the computational elements.
# We will visualize these effects by recording the membrane potential of
# multiple neurons in leak-over-threshold configuration. You will notice
# different resting, reset and threshold potentials as well as varying
# membrane time constants.
plt.figure()
plt.suptitle("Third experiment: Fixed-pattern variations")
pynn.setup()

pop = pynn.Population(10, pynn.cells.HXNeuron(
    # Leak potential, range: 300-1000
    leak_v_leak=1000,
    # Leak conductance, range: 0-1022
    leak_i_bias=200,
    # Threshold potential, range: 0-600
    threshold_v_threshold=300,
    # Reset potential, range: 300-1000
    reset_v_reset=400,
    # Membrane capacitance, range: 0-63
    membrane_capacitance_capacitance=63,
    # Refractory time, range: 0-255
    refractory_period_refractory_time=120,
    # Enable reset on threshold crossing
    threshold_enable=True,
    # Reset conductance, range: 0-1022
    reset_i_bias=1022,
    # Enable strengthening of reset conductance
    reset_enable_multiplication=True))

for neuron_id in range(len(pop)):
    logger.INFO(f"Recording fixed-pattern variations: Run {neuron_id}")
    p_view = pynn.PopulationView(pop, [neuron_id])
    p_view.record(["v"])
    pynn.run(0.1)
    plot_membrane_dynamics(p_view)
    pynn.reset()
    pop.record(None)

# Show the recorded membrane traces of multiple different neurons. Due to
# the time-continuous nature of the system, there is no temporal alignment
# between the individual traces, so the figure shows multiple independent
# effects:
#  * Temporal misalignment: From the system's view, the recording happens in
#    an arbitrary time frame during the continuously evolving integration.
#    Neurons are not synchronized to each other.
#  * Circuit-level mismatch: Each individual neurons shows slightly
#    different analog properties. The threshold is different for all traces;
#    as is the membrane time constant (visible as slope) and the reset
#    potentials (visible as plateaus during the refractory time).
plt.show()
pynn.end()

# -- Forth experiment: External stimulation -- #
# Up to now, we have observed analog neurons without external stimulus. In
# this experiment, we will introduce the latter and examine post-synaptic
# pulses on the analog neuron's membrane.
plt.figure()
plt.suptitle("Forth experiment: External stimulation")
pynn.setup()

# Preparing the neuron to receive synaptic inputs requires the configuration
# of additional circuits. The additional settings include technical parameters
# for bringing the circuit to its designed operating point as well as
# configuration with a direct biological equivalent.
stimulated_p = pynn.Population(1, pynn.cells.HXNeuron(
    # Leak potential, range: 300-1000
    leak_v_leak=400,
    # Leak conductance, range: 0-1022
    leak_i_bias=200,
    # Threshold potential, range: 0-600
    threshold_v_threshold=400,
    # Reset potential, range: 300-1000
    reset_v_reset=300,
    # Membrane capacitance, range: 0-63
    membrane_capacitance_capacitance=63,
    # Refractory time, range: 0-255
    refractory_period_refractory_time=120,
    # Enable reset on threshold crossing
    threshold_enable=True,
    # Reset conductance, range: 0-1022
    reset_i_bias=1022,
    # Enable strengthening of reset conductance
    reset_enable_multiplication=True,

    # -- Parameters for synaptic inputs -- #
    # Enable synaptic stimulation
    excitatory_input_enable=True,
    inhibitory_input_enable=True,
    # Strength of synaptic inputs
    excitatory_input_i_bias_gm=1022,
    inhibitory_input_i_bias_gm=1022,
    # Synaptic time constants
    excitatory_input_i_bias_tau=200,
    inhibitory_input_i_bias_tau=200,
    # Technical parameters
    excitatory_input_i_drop_input=300,
    inhibitory_input_i_drop_input=300,
    excitatory_input_i_shift_reference=300,
    inhibitory_input_i_shift_reference=300))

stimulated_p.record(["v", "spikes"])

# Create off-chip populations serving as excitatory external spike sources
exc_spiketimes = [0.01, 0.05, 0.07, 0.08]
exc_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=exc_spiketimes))

# We represent projections as entries in the synapse matrix on the neuromorphic
# chip. Weights are stored in digital 6bit values (plus sign), the value
# range for on-chip weights is therefore -63 to 63.
# With this first projection, we connect the external spike source to the
# observed on-chip neuron population.
pynn.Projection(exc_stim_pop, stimulated_p,
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=63),
                receptor_type="excitatory")

# Create off-chip populations serving as inhibitory external spike sources.
inh_spiketimes = [0.03]
inh_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=inh_spiketimes))

pynn.Projection(inh_stim_pop, stimulated_p,
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=-63),
                receptor_type="inhibitory")

# You may play around with the parameters in this experiment to achieve
# different traces. Try to stack multiple PSPs, try to make the neurons spike,
# try to investigate differences between individual neuron instances,
# be creative!

pynn.run(0.1)
plot_membrane_dynamics(stimulated_p)
plt.show()
