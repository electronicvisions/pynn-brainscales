from typing import Optional, Union, List

import numpy as np
import pyNN.common
from pyNN.parameters import ParameterSpace, simplify
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.plasticity_rules import PlasticityRuleHandle
from pynn_brainscales.brainscales2.recording import Recorder
import pygrenade_vx.network.abstract.frontend as grenade
from neo.io import NixIO


class Assembly(pyNN.common.Assembly):
    _simulator = simulator

    # Note: Forward `device` argument. Rest of implementation identical to
    # upstream PyNN.
    def record(self, variables, to_file=None, sampling_interval=None,
               locations=None, *, device="madc"):
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
        `device` defines the device to use.
        """
        for pop in self.populations:
            pop.record(variables, to_file, sampling_interval,
                       locations=locations, device=device)


# pylint:disable=abstract-method
class Population(pyNN.common.Population, grenade.ExperimentElement):
    __doc__ = pyNN.common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly
    all_cells: np.ndarray
    _mask_local: np.ndarray
    _description_cache = {
        # hash of describe(…) call arguments -> return value
    }

    def __init__(self, *args, **kwargs):
        grenade.ExperimentElement.__init__(
            self, self._simulator.state.grenade_experiment)
        self.grenade_descriptor = None
        super().__init__(*args, **kwargs)

    def _create_cells(self):
        id_range = np.arange(self._simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)
        self._mask_local = np.ones((self.size,), bool)  # all cells are local

        for cell_id in self.all_cells:
            cell_id.parent = self

        parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        # we want to iterate over parameter space -> evaluate to get rid of
        # laziness
        parameter_space.evaluate(mask=self._mask_local, simplify=False)

        simulator.state.id_counter += self.size
        simulator.state.populations.append(self)

    def _get_parameters(self, *names):
        """Return a ParameterSpace containing native parameters"""
        self.celltype.validate_parameter_space()
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
        self.celltype.validate_parameter_space()
        self.changed_input_data = True
        self._description_cache.clear()
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            # Since pyNN 0.12.0 a LazyArray is generated when we assign
            # a new value with parameterspace[name] = value. As a result
            # the parmeter space is no longer consistently "evaluated"
            # (since a lazy array is saved) -> assign value directly to
            # private member
            # TODO: Discuss if upstream PyNN implementation is correct/wanted.
            self.celltype.parameter_space._parameters[name] = value  # pylint: disable=protected-access

    def __setattr__(self, name, value):
        # Handle (de-)registering of population in plasticity rule.
        # A plasticity rule can be applied to multiple populations
        # and then serve handles to all in the kernel code, for which
        # registration here is required.
        self._description_cache.clear()
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

    # pylint: disable=unused-argument
    def _set_initial_value_array(self, variable, value):
        return

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def describe(self, *args, **kwargs):
        # We have a lot of parameters -> super().describe(…) is slow, but may
        # be called often.
        call_args = args + tuple(sorted(kwargs.items()))
        argument_hash = hash(call_args)

        if argument_hash not in self._description_cache:
            self._description_cache[argument_hash] = \
                super().describe(*args, **kwargs)

        return self._description_cache[argument_hash]

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

        if self.celltype.plasticity_rule._recording_data is None:  # pylint: disable=protected-access
            raise RuntimeError(
                "Plasticity rule observables only available after execution.")
        observable_data = []
        for snippet in self.celltype.plasticity_rule._recording_data:  # pylint: disable=protected-access
            if snippet is not None and observable in snippet.data_per_neuron:
                observable_data.append(
                    snippet.data_per_neuron[observable][
                        self.celltype.plasticity_rule._populations.index(  # pylint: disable=protected-access
                            self)][
                        simulator.state.batch_entry])
            else:
                observable_data.append(None)

        return observable_data

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

    # Note: Forward `device` argument. Rest of implementation identical to
    # upstream PyNN.
    def record(self,
               variables: Union[str, List[str]],
               to_file: Optional[Union[str, NixIO]] = None,
               sampling_interval: Optional[float] = None,
               locations: Optional[List[str]] = None,
               *,
               device: str = "madc") -> None:

        """
        Record the specified variable or variables for all cells in the
        Population or view.

        :param variables: may be either a single variable name or a list of
                variable names. For a given celltype class,
                `celltype.recordable` contains a list of variables that can be
                recorded for that celltype. To reset recording of
                PopulationView or Population call `record` with "None" as
                argument.

        :param to_file: If specified, variables are written
                to file when `pynn.end()` is called. For more details see
                :py:function::pynn.population.write_data or pyNN-Doc
                <https://neuralensemble.org/docs/PyNN/data_handling.html>

        :param sampling_interval: Feature not yet implemented.

        :param locations: Defines where the variables should be recorded. Used
                with multi-compartment neurons where location is defined by
                `label`.

        :param device: Configures device which is used to record recordables.
                Default is set to "madc". Use as device
                "pad_[0/1]_[unbuffered/buffered]" to connect recordables of
                neurons to pads. Each of the 2 pads can be connected to one
                neuron. For limitations see documentation.
        """
        # we currently directly change the parameterization of the neuron
        # circuits for the HXNeuron and the calibrated neurons to change
        # the readout source -> we need to regenerate the input data when
        # changing the recording
        self.changed_input_data = True

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
                                     locations=locations,
                                     device=device)
            else:
                self.recorder.record(variables, self._record_filter,
                                     sampling_interval,
                                     locations=locations,
                                     device=device)
        if isinstance(to_file, str):
            self.recorder.file = to_file
            self._simulator.state.write_on_end.append((self, variables,
                                                      self.recorder.file))

    # TODO: align handling of varibales to PyNN 0.12
    # this function was just copied from PyNN 0.10.1
    def find_units(self, variable):
        return self.celltype.units[variable]

    def add_to_topology(
            self,
            experiment: grenade.ExperimentSnippet):
        if self.grenade_descriptor is not None and \
                experiment.topology.contains(self.grenade_descriptor):
            experiment.topology.set(
                self.grenade_descriptor,
                self.celltype.generate_vertex(self))
        else:
            self.grenade_descriptor = experiment.topology.add_vertex(
                self.celltype.generate_vertex(self))
        return True

    def add_to_input_data(
            self,
            experiment: grenade.ExperimentSnippet,
            snippet_begin_time,
            snippet_end_time):
        input_data = self.celltype.generate_input_data(
            self, experiment, snippet_begin_time, snippet_end_time)
        if input_data is None:
            return
        for port_on_vertex, port_data in input_data.items():
            experiment.input_data.ports.set(
                (self.grenade_descriptor, port_on_vertex), port_data)

    def extract_output_data(
            self,
            experiment: List[grenade.ExperimentSnippet]):
        pass


# pylint:disable=abstract-method
class PopulationView(pyNN.common.PopulationView):
    _simulator = simulator
    _assembly_class = Assembly

    def _get_parameters(self, *names):
        """Return a ParameterSpace containing native parameters"""
        self.celltype.validate_parameter_space()
        parameter_space = {}
        for name in names:
            value = simplify(self.celltype.parameter_space[name])
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_space[name] = value
        # need to add schema such that lazyarrays in the parameter space get
        # a `dtype` assigned and can be evaluated later
        schema = {key: value for key, value in
                  self.celltype.get_schema().items() if key in names}
        return ParameterSpace(parameter_space, schema=schema,
                              shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        self.celltype.validate_parameter_space()
        self.parent.changed_input_data = True
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            self.celltype.parameter_space[name][self.mask] = value

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    # Accessors for automatic calibration neuron types
    @property
    def actual_hwparams(self):
        try:
            return np.array(self.celltype.actual_hwparams)[self.mask]
        except AttributeError:
            # pylint:disable=raise-missing-from
            raise AttributeError("actual_hwparams not available for celltype "
                                 f"{type(self.celltype)}")

    @property
    def calib_hwparams(self):
        try:
            return np.array(self.celltype.calib_hwparams)[self.mask]
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
