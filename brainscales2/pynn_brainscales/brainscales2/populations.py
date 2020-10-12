import numpy as np
import pyNN.common
from pyNN.parameters import ParameterSpace
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.recording import Recorder


class PopulationMixin:

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """Return a ParameterSpace containing native parameters"""
        parameter_space = {}
        for name in names:
            parameter_space.update(
                {name: self.initial_values[name].base_value})

        # shape must be declared, the actual value doesn't seem to have an
        # effect, though
        return ParameterSpace(parameter_space, shape=(1,))

    # TODO: cf. feature #3611
    def _set_parameters(self, parameter_space):
        raise NotImplementedError


class Assembly(pyNN.common.Assembly):
    _simulator = simulator


# pylint:disable=abstract-method
class Population(pyNN.common.Population, PopulationMixin):
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly
    not_configurable = ["event_routing_analog_output",
                        "event_routing_enable_digital",
                        "leak_reset_i_bias_source_follower",
                        "readout_enable_amplifier",
                        "readout_source",
                        "readout_enable_buffered_access",
                        "readout_i_bias"]

    # pylint: disable=too-many-arguments
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values=None, label=None):
        """
        Create a population of neurons all of the same type.

        :param size: number of cells in the Population. For backwards-
                     compatibility, `size` may also be a tuple giving the
                     dimensions of a grid, e.g. ``size=(10,10)`` is
                     equivalent to ``size=100`` with ``structure=Grid2D()``.

        :param cellclass: a cell type (a class inheriting from
                          :class:`pyNN.models.BaseCellType`).

        :param cellparams: a dict, or other mapping, containing parameters,
                           which is passed to the neuron model constructor,
                           defaults to None

        :param structure: a :class:`pyNN.space.Structure` instance, used to
                          specify the positions of neurons in space,
                          defaults to None

        :param initial_values: a dict, or other mapping, containing initial
                               values for the neuron state variables,
                               defaults to None

        :param label: a name for the population (one will be auto-generated
                      if this is not supplied),
                      defaults to None
        """

        if cellparams is None:
            cellparams = {}

        if initial_values is None:
            initial_values = {}

        for key, _ in initial_values.items():
            if key in self.not_configurable:
                raise ValueError("{} is not configurable.".format(key))

        self.size = size
        id_range = np.arange(self._simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)
        self._mask_local = np.ones((self.size,), bool)  # all cells are local

        super(Population, self).__init__(size, cellclass, cellparams,
                                         structure, initial_values, label)
        self._simulator.state.populations.append(self)

    def _create_cells(self):
        for cell_id in self.all_cells:
            cell_id.parent = self

        simulator.state.id_counter += self.size

    # pylint: disable=unused-argument, no-self-use
    def _set_initial_value_array(self, variable, value):
        return


# pylint:disable=abstract-method
class PopulationView(pyNN.common.PopulationView, PopulationMixin):
    _simulator = simulator
    _assembly_class = Assembly

    @property
    def initial_values(self):
        return self.parent.initial_values
