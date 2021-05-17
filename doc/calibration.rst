Calibration
===========
BrainScaleS-2 uses analog circuits to emulate the behavior of neurons and synapses.
Due to the manufacturing process these circuits are subject to variations.
This means that every analog circuit has slightly different properties.

In order to reduce the mismatch between different neurons and synapses the BrainScaleS-2 system can be calibrated.
The `calix` library is capable of performing these calibration.
In this document we describe how calibration data can be used in PyNN.

.. currentmodule:: pynn_brainscales.brainscales2

Loading a Calibration
---------------------
The helper function :py:func:`~helper.chip_from_file` converts a binary dump of a playback program to an object which represents the configuration of the BrainScaleS-2 chip.

.. code:: python3

    import pynn_brainscales.brainscales2 as pynn
    chip = pynn.helper.chip_from_file(some_file)


Applying a Calibration
^^^^^^^^^^^^^^^^^^^^^^
The global chip configuration can be injected when we call the :py:func:`~setup` function.

.. code:: python3

    pynn.setup(initial_config=chip)


Nightly Calibration
^^^^^^^^^^^^^^^^^^^
Every night a default calibration is generated for each setup.
The path to the most recent calibration for the setup in use can be fetched with:

.. autofunction:: pynn_brainscales.brainscales2.helper.nightly_calib_path
   :noindex:

Furthermore, a convenient function is provided which allows to directly retrieve the chip configuration:

.. autofunction:: pynn_brainscales.brainscales2.helper.chip_from_nightly
   :noindex:


Generating the Calibration from the PyNN Network
------------------------------------------------
In addition to loading a previously created calibration into PyNN, the calibration can also be directly be created for a network defined in PyNN.
This feature can be used by replacing the :py:class:`~standardmodels.cells.HXNeuron` by one of the following cell types:

.. autoclass:: pynn_brainscales.brainscales2.standardmodels.cells.CalibHXNeuronCuba
   :noindex:

.. autoclass:: pynn_brainscales.brainscales2.standardmodels.cells.CalibHXNeuronCoba
   :noindex:

Once :py:func:`~run` is called, `calix` is used to generate a calibration in the background.
As generating a calibration takes around five minutes, this mode is not meant for interactive exploration.
