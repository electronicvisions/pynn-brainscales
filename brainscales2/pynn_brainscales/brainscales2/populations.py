import numpy as np
import pyNN.common
from pyNN.parameters import ParameterSpace, simplify
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.recording import Recorder


class Assembly(pyNN.common.Assembly):
    _simulator = simulator


# pylint:disable=abstract-method
class Population(pyNN.common.Population):
    __doc__ = pyNN.common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly
    all_cells: np.ndarray
    _mask_local: np.ndarray
    changed_since_last_run = True

    def _create_cells(self):
        id_range = np.arange(self._simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)
        self._mask_local = np.ones((self.size,), bool)  # all cells are local

        for cell_id in self.all_cells:
            cell_id.parent = self

        if hasattr(self.celltype, "create_hw_entity"):
            simulator.state.neuron_placement.register_id(self.all_cells)
            coords = simulator.state. \
                neuron_placement.id2atomicneuron(self.all_cells)
            self.celltype.apply_config(coords)

        parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        # we want to iterate over parameter space -> evaluate to get rid of
        # laziness
        parameter_space.evaluate(mask=self._mask_local, simplify=False)

        simulator.state.id_counter += self.size
        simulator.state.populations.append(self)

    def _get_parameters(self, *names):
        """Return a ParameterSpace containing native parameters"""
        parameter_space = {}
        for name in names:
            value = self.celltype.parameter_space[name]
            value = simplify(value)
            # lazyarray cannot handle np.bool_
            if isinstance(value, np.bool_):
                value = bool(value)
            parameter_space[name] = value
        return ParameterSpace(parameter_space, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        self.changed_since_last_run = True
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            self.celltype.parameter_space[name] = value

    # pylint: disable=unused-argument, no-self-use
    def _set_initial_value_array(self, variable, value):
        return

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


# pylint:disable=abstract-method
class PopulationView(pyNN.common.PopulationView):
    _simulator = simulator
    _assembly_class = Assembly

    def _get_parameters(self, *names):
        """Return a ParameterSpace containing native parameters"""
        parameter_space = {}
        for name in names:
            value = simplify(self.celltype.parameter_space[name])
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_space[name] = value
        return ParameterSpace(parameter_space, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        self.parent.changed_since_last_run = True
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            self.celltype.parameter_space[name][self.mask] = value

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)
