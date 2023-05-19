import numpy as np
import pyNN.common
from pyNN.parameters import ParameterSpace, simplify
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.plasticity_rules import PlasticityRuleHandle
from pynn_brainscales.brainscales2.recording import Recorder


class Assembly(pyNN.common.Assembly):
    _simulator = simulator

    # Note: Forward `location` argument. Rest of implementation identical to
    # upstream PyNN.
    def record(self, variables, to_file=None, sampling_interval=None,
               locations=None):
        """
        Record the specified variable or variables for all cells in the
        Assembly.

        `variables` may be either a single variable name or a list of variable
        names. For a given celltype class, `celltype.recordable` contains a
        list of variables that can be recorded for that celltype.

        If specified, `to_file` should be either a filename or a Neo IO
        instance and `write_data()` will be automatically called when `end()`
        is called.

        `locations` defines where the variables should be recorded.
        """
        for pop in self.populations:
            pop.record(variables, to_file, sampling_interval, locations)


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

        if hasattr(self.celltype, "logical_compartments"):
            simulator.state.neuron_placement.register_neuron(
                self.all_cells, self.celltype.logical_compartments)
            if hasattr(self.celltype, "apply_config"):
                coords = simulator.state.neuron_placement.\
                    id2logicalneuron(self.all_cells)
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
        # need to add schema such that lazyarrays in the parameter space get
        # a `dtype` assigned and can be evaluated later
        schema = {key: value for key, value in
                  self.celltype.get_schema().items() if key in names}
        return ParameterSpace(parameter_space, schema=schema,
                              shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        self.changed_since_last_run = True
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            self.celltype.parameter_space[name] = value

    def __setattr__(self, name, value):
        # Handle (de-)registering of population in plasticity rule.
        # A plasticity rule can be applied to multiple populations
        # and then serve handles to all in the kernel code, for which
        # registration here is required.
        if name == "celltype":
            if hasattr(self, name):
                if isinstance(self.celltype, PlasticityRuleHandle) \
                        and self.celltype.plasticity_rule is not None:
                    self.celltype.plasticity_rule._remove_population(self)
            super().__setattr__(name, value)
            if isinstance(self.celltype, PlasticityRuleHandle) \
                    and self.celltype.plasticity_rule is not None:
                self.celltype.plasticity_rule._add_population(self)
        else:
            super().__setattr__(name, value)

    # pylint: disable=unused-argument, no-self-use
    def _set_initial_value_array(self, variable, value):
        return

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def get_plasticity_data(self, observable: str):
        """
        Get recorded observable data for this population from plasticity
        rule.
        :raises RuntimeError: On no plasticity rule or requested observable
            name present
        :param observable: Observable name to get data for
        :returns: Observable data per plasticity rule execution period per
            neuron
        """
        if not isinstance(self.celltype, PlasticityRuleHandle):
            raise RuntimeError("Celltype can't have observables, since it"
                               + " is not derived from PlasticityRuleHandle.")
        if self.celltype.plasticity_rule is None:
            raise RuntimeError("Celltype can't have observables, since it"
                               + " does not hold a plasticity rule.")
        if observable not in self.celltype.plasticity_rule.observables:
            raise RuntimeError(
                "Celltype doesn't have requested observable.")
        return self._simulator.state.neuronal_observables[
            self._simulator.state.populations.index(self)][observable]

    # Accessors for automatic calibration neuron types
    @property
    def actual_hwparams(self):
        try:
            return self.celltype.actual_hwparams
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("actual_hwparams not available for celltype "
                                 f"{type(self.celltype)}")

    @property
    def calib_hwparams(self):
        try:
            return self.celltype.calib_hwparams
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("calib_hwparams not available for celltype "
                                 f"{type(self.celltype)}")

    @property
    def calib_target(self):
        try:
            return self.celltype.calib_target
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("calib_target not available for celltype "
                                 f"{type(self.celltype)}")

    # Note: Forward `location` argument. Rest of implementation identical to
    # upstream PyNN.
    def record(self, variables, to_file=None, sampling_interval=None,
               locations=None):
        """
        Record the specified variable or variables for all cells in the
        Population or view.

        `variables` may be either a single variable name or a list of variable
        names. For a given celltype class, `celltype.recordable` contains a
        list of variables that can be recorded for that celltype.

        If specified, `to_file` should be either a filename or a Neo IO
        instance and `write_data()` will be automatically called when `end()`
        is called.

        `sampling_interval` should be a value in milliseconds, and an integer
        multiple of the simulation timestep.

        `locations` defines where the variables should be recorded.
        """
        if variables is None:  # reset the list of things to record
            # note that if record(None) is called on a view of a population
            # recording will be reset for the entire population, not just the
            # view
            self.recorder.reset()
        else:
            self._simulator.state.log.debug("%s.record('%s')",
                                            self.label, variables)
            if self._record_filter is None:
                self.recorder.record(variables, self.all_cells,
                                     sampling_interval,
                                     locations)
            else:
                self.recorder.record(variables, self._record_filter,
                                     sampling_interval,
                                     locations)
        if isinstance(to_file, str):
            self.recorder.file = to_file
            self._simulator.state.write_on_end.append((self, variables,
                                                      self.recorder.file))


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

    # Accessors for automatic calibration neuron types
    @property
    def actual_hwparams(self):
        try:
            return self.celltype.actual_hwparams[self.mask]
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("actual_hwparams not available for celltype "
                                 f"{type(self.celltype)}")

    @property
    def calib_hwparams(self):
        try:
            return self.celltype.calib_hwparams[self.mask]
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("calib_hwparams not available for celltype "
                                 f"{type(self.celltype)}")

    @property
    def calib_target(self):
        try:
            return self.celltype.calib_target[self.mask]
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("calib_target not available for celltype "
                                 f"{type(self.celltype)}")

    record = Population.record
