.. _pynn-brainscales:


PyNN for BrainScaleS-2
======================
.. module:: pynn_brainscales

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Content

   hx_neuron
   calibration

BrainScaleS-2 allows users to use the `PyNN API <http://neuralensemble.org/docs/PyNN/introduction.html>`_ to define experiments of spiking neural network.
This documentation provides details to the BrainScaleS-2 implementation of PyNN and highlights differences to the standard PyNN interface.
More details to the PyNN API can be found in the corresponding `documentation <http://neuralensemble.org/docs/PyNN/index.html>`_.

BrainScaleS-2 is an accelerated, mixed-signal neuromorphic chip; its analog circuits implement the dynamics of the adaptive exponential integrate-and-fire neuron model.
The custom cell type :py:class:`~brainscales2.standardmodels.cells.HXNeuron` allows to set the "hardware parameters" of neuron circuits directly.
For more information about the :py:class:`~pynn_brainscales.brainscales2.standardmodels.cells.HXNeuron` see :doc:`the corresponding documentation <hx_neuron>`.

In addition the cell types :py:class:`~brainscales2.standardmodels.cells.SpikeSourceArray` and :py:class:`~brainscales2.standardmodels.cells.SpikeSourcePoisson` are available to inject external spikes into the network.
Just like in standard PyNN, populations of neurons and projections between populations are used to define the network architecture.
A good starting point to get familiar with BrainScales2 and its PyNN interface are the :doc:`demos and tutorials</brainscales2-demos/index>`.

Before the network is emulated on the BrainSacaleS-2 system, the abstract network description has to be translated to a valid hardware configuration.
This mapping is performed by :doc:`grenade </api_grenade>`

Recording and receiving of observables such as spikes and membrane voltages work as in standard PyNN.
For a list of recordable observables refer to the :doc:`HX neuron documentation<hx_neuron>`.

API Overview
------------

An overview over the full API can be found in :doc:`/api_pynn-brainscales2`.
