The HXNeuron
============

BrainScaleS-2 emulates the dynamics of the adaptive exponential integrate-and-fire neuron model.
In order to configure neurons on BrainScaleS-2 the custom neuron model :py:class:`~pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron` is introduced.

.. autoclass:: pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron
   :noindex:

Parameters
----------

The :py:class:`~pynn-brainscales2.brainscales2.standardmodels.cells.HXNeuron` allows access to a subset of the :cpp:class:`lola::vx::v2::AtomicNeuron`.
Parameters which are related to routing of spike events or the recording of voltages are not exposed as their configuration is handeled by internal logic.
All relevant analog as well as digital values of the underlying neuron circuit are exposed and can be configured.
As descriped in the doc string of :py:class:`~pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron` the parameter hierarchy of the :cpp:class:`lola::vx::v2::AtomicNeuron` is flattened.
For example accessing the leak conductance in python works as follows:

.. code:: python3

    lola_neuron = lola.AtomicNeuronOnDLS()
    lola_neuron.leak.v_leak = 200

    neuron = pynn.Population(1, HXNeuron(leak_v_leak=200))


A detailed example of how to configure a :py:class:`~pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron` is given in the demo ":doc:`/brainscales2-demos/ts_00-single_neuron.rst`".

Observables
-----------

The :py:class:`~pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron` allows to record spikes as well as different state variables.

Spikes
^^^^^^
Spikes can be recorded in parallel for all neurons. The syntax for recording spikes works as follows:

.. code:: python3

    population = pynn.Population(10, pynn.cells.HXNeuron())
    population.record('spikes')


State Variables
^^^^^^^^^^^^^^^

Two analog-to-digital converters (ADCs) are available on the BrainScaleS-2 neuromorphic chip.
The PyNN interface of  BrainScaleS-2 allows to record two state variables of two different neurons at the same time.
On harware, even neuron circuits are connected to one readout line and odd neurons to another readout line.
As a result, one even and one odd neuron can be recorded at the same time (but not two even or two odd neurons).
The following state variables are available for recording:

.. autoattribute:: pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron.recordable
   :noindex:

For example the membrane voltage and spikes can be recorded in the standard PyNN way:

.. code:: python3

    population = pynn.Population(1, pynn.cells.HXNeuron())
    population.record(['v', 'spikes'])
