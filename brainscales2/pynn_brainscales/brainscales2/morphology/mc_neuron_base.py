from abc import ABC
from collections.abc import Iterable
from copy import deepcopy
import numbers
from typing import List, Final, Dict, Any, Sequence

import numpy as np

from pyNN.common import Population
from pyNN.standardmodels import build_translations, StandardCellType

from dlens_vx_v3 import lola, halco, hal
import pygrenade_vx.network as grenade

from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.standardmodels.cells_base import \
    NetworkAddableCell
from pynn_brainscales.brainscales2.recording import RecordingSite
from pynn_brainscales.brainscales2.helper import decompose_in_member_names, \
    get_values_of_atomic_neuron
from pynn_brainscales.brainscales2.morphology.parameters import \
    McCircuitParameters


def _expand_to_size(value: Any, size: int) -> Iterable:
    '''
    Return iterable with the given size.

    Return `value` if value already has the desired `size` otherwise
    return list with `value` repeated `size` times.

    Examples:
    >>> _expand_to_size(1, 3)
    [1, 1, 1]

    >>> _expand_to_size([1, 2], 3)
    [[1, 2], [1, 2], [1, 2]]

    >>> _expand_to_size([1, 2, 3], 3)
    [1, 2, 3]

    :param value: Value to expand.
    :param size: Size of iterable which should be returned.
    :return: Iterable with the desired size.
    '''
    if not isinstance(value, Iterable) or len(value) != size:
        return [value] * size
    return value


class McNeuronBase(StandardCellType, NetworkAddableCell, ABC):
    '''
    Base class for multi-compartmental neuron models.

    Contains the core functionality for multi-compartment neurons.
    It flattens the hierarchy of a `lola.AtomicNeuron`, for example
    `lola.AtomicNeuron.multicompartment.i_bias_nmda` is expressed as the
    parameter `multicompartment_i_bias_nmda`.
    Parameters related to the event routing, configuration of morphology and
    readout are not exposed since they are determined by other settings in
    PyNN.

    A subclass is expected to set the following member variables:
        - compartments: Dictionary of Compartment Ids and Compartments.
        - logical_neuron: Configuration of the neuron in form of a logical
                          neuron.
        - logical_compartments: Shape of the morphology.

    '''
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]
    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = False
    injectable: Final[bool] = True

    # the actual unit of `v` is `haldls::vx::CapMemCell::Value`
    # [0–1022]; 1023 means off,
    # but only units included in the `quantity` package are accepted
    units: Final[Dict[str, str]] = {"v": "dimensionless",  # pylint: disable=invalid-name
                                    "exc_synin": "dimensionless",
                                    "inh_synin": "dimensionless",
                                    "adaptation": "dimensionless"}

    _NON_CONFIGURABLE: Final[List[str]] = [
        "event_routing_analog_output",
        "event_routing_enable_digital",
        "leak_reset_i_bias_source_follower",
        "multicompartment_connect_soma",
        "multicompartment_connect_soma_right",
        "multicompartment_connect_right",
        "multicompartment_connect_vertical",
        "multicompartment_enable_conductance",
        "multicompartment_enable_conductance_division",
        "multicompartment_enable_conductance_multiplication",
        "readout_enable_amplifier",
        "readout_source",
        "readout_enable_buffered_access",
        "readout_i_bias"]

    # the following class members have to be implemented by a subclass
    compartments: Dict = {}
    logical_neuron: lola.LogicalNeuron = lola.LogicalNeuron()
    logical_compartments: halco.LogicalNeuronCompartments = \
        halco.LogicalNeuronCompartments()

    # only use first circuit for leak, capacitance and spiking
    single_active_circuit = True

    def __init__(self, **parameters):
        self._user_provided_parameters = parameters
        super().__init__(**parameters)

    # check that members are correctly implemented
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        n_comp = len(cls.compartments)
        if n_comp == 0:
            raise RuntimeError(
                'A multi-compartment neuron class has to have at least one '
                'compartment. Did you implement `cls.compartments`?')

        # there might be more compartments in the logical neuron since some
        # neurons might only be used to connect different compartments, i.e.
        # the somatic line is closed. These neurons are assigned to
        # compartments in the logical neuron
        n_comp_logical = len(cls.logical_neuron.morphology)
        if n_comp_logical < n_comp:
            raise RuntimeError(
                'There is a mismatch between the number of compartments in '
                '`cls.compartments` and `cls.logical_neuron`. Did you '
                'implement `cls.logical_neuron`?')

        n_comp_coord = len(cls.logical_compartments.get_compartments())
        if n_comp_coord != n_comp_logical:
            raise RuntimeError(
                'There is a mismatch between the number of compartments in '
                '`cls.logical_neuron` and `cls.logical_compartments`. Did you '
                'implement `cls.logical_compartments`?')

        cls._update_default_parameters()

    @staticmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        # get neuron coordinates
        coords: List[halco.LogicalNeuronOnDLS] = \
            simulator.state.neuron_placement.id2logicalneuron(
                population.all_cells)

        # create receptors
        receptors = set([
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.excitatory),
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.inhibitory),
        ])

        neurons = []
        for cell_id, coord in zip(population.all_cells, coords):
            comps = {}
            for comp_id, comp in population.celltype.compartments.items():
                record = RecordingSite(cell_id, comp_id) in \
                    population.recorder.recorded["spikes"]
                spike_master = grenade.Population.Neuron\
                    .Compartment.SpikeMaster(0, record)
                comps[comp_id] = grenade.Population.Neuron\
                    .Compartment(spike_master, [receptors] * comp.size)

            neurons.append(grenade.Population.Neuron(coord, comps))
        # create grenade population
        gpopulation = grenade.Population(neurons)
        # add to builder
        descriptor = builder.add(gpopulation)

        return descriptor

    @staticmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):
        pass

    @classmethod
    def get_compartment_ids(cls, labels: Sequence[str]
                            ) -> List[halco.CompartmentOnLogicalNeuron]:
        '''
        Get compartment Ids of compartments with the specified labels.

        :param labels: Labels for which to extract the compartment IDs.
        :return: IDs of compartments for which the given labels match.
            Note that a single label can match one or more compartment IDS.
        :raises ValueError: If no compartment can be matched to one of the
            given labels.
        '''
        all_labels = np.asarray(cls.get_labels())

        def label_to_ids(label: str) -> np.ndarray:
            ids = np.where(all_labels == label)[0]
            if len(ids) == 0:
                raise ValueError(f'Label "{label}" does not exist.')
            return ids

        indices = np.array([label_to_ids(label) for label in labels]).flatten()
        return [halco.CompartmentOnLogicalNeuron(index) for index in indices]

    @classmethod
    def get_labels(cls) -> List[str]:
        '''
        Retrieve list of all used labels.

        :return: List of all used labels.
        '''
        return [value.label for value in cls.compartments.values()]

    @classmethod
    def get_label(cls, compartment_id: halco.CompartmentOnLogicalNeuron
                  ) -> str:
        '''
        Retrieve label of a single compartment.

        :param compartment_id: ID of compartment for which to retrieve the
            label.
        :return: Label of selected compartment.
        '''
        return cls.compartments[compartment_id].label

    @classmethod
    def get_default_values(cls) -> dict:
        return get_values_of_atomic_neuron(lola.AtomicNeuron(),
                                           cls._NON_CONFIGURABLE)

    @classmethod
    def _update_default_parameters(cls) -> None:
        """
        Determine the default values for each neuron parameter and update
        class variable `default_parameters`.

        Overwrite default values of parameters (given by
        `cls.get_default_values`) with values given for the different
        compartments. Expand the size of each parameter to fit the number
        of compartments and number of neuron circuits in each compartment.
        """
        circuit_defaults = cls.get_default_values()

        default_values = {}
        for name, default_value in circuit_defaults.items():
            comp_values = []
            for comp in cls.compartments.values():
                if name in comp.parameters:
                    circuit_values = _expand_to_size(comp.parameters[name],
                                                     comp.size)
                else:
                    circuit_values = _expand_to_size(default_value, comp.size)
                comp_values.append(circuit_values)
            default_values[name] = McCircuitParameters(comp_values)

        if cls.single_active_circuit:
            def change_value(name, new_value):
                default_values[name] = cls._change_all_but_first_circuit(
                    default_values[name], new_value)

            change_value('leak_i_bias', 0)
            change_value('leak_enable_division', True)
            change_value('leak_enable_multiplication', False)
            change_value('membrane_capacitance_capacitance', 0)
            change_value('threshold_enable', False)

        cls.default_parameters = default_values

    @classmethod
    def _create_translation(cls) -> dict:
        default_values = cls.get_default_values()
        translation = []
        for key in default_values:
            translation.append((key, key))

        return build_translations(*translation)

    def get_logical_neuron(self, cell_in_pop: int) -> lola.LogicalNeuron:
        '''
        Extract parameters from parameter space and apply it to a logical
            neuron.

        :param cell_in_pop: Index of neuron in population for which to return
            the configuration as a logical neuron.
        :return: Configuration of a single neuron in form of a logical neuron.
        '''

        # base configuration
        config = self.logical_neuron

        # apply settings from parameter space
        for name, values in self.parameter_space.items():
            # extract member variable of McSafeAtomicNeuron and the name
            # of its member variable (separated by the first underscore)
            atomic_member_name, member_name = decompose_in_member_names(name)
            value = values[cell_in_pop]
            for comp_id, circuits in config.morphology.items():  # pylint: disable=no-member
                for circuit_id, circuit in enumerate(circuits):
                    try:
                        sing_val = value.value[comp_id][circuit_id]
                    except AttributeError:
                        sing_val = value

                    if isinstance(sing_val, numbers.Real):
                        # CapMem values have to be integers
                        sing_val = int(sing_val)
                    atomic_member = getattr(circuit, atomic_member_name)
                    val = type(getattr(atomic_member, member_name))(sing_val)
                    setattr(atomic_member, member_name, val)

        return config

    def _add_morphology_params_to_ps(self) -> None:
        '''
        Overwrite values in the parameter space with parameters saved for the
        individual compartments.

        Parameters specified during the creation of the morphology should take
        precedence over values read from calibration. Therefore, these values
        are extracted from the individual compartments and the parameter space
        is updated accordingly.
        '''

        for n_comp, comp in self.compartments.items():
            for param_name, comp_param in comp.parameters.items():

                # expand compartment parameter to correct size
                if not isinstance(comp_param, Iterable) or \
                        isinstance(comp_param, str):
                    comp_param = np.array([comp_param] * comp.size)
                elif len(comp_param) != comp.size:
                    raise RuntimeError(
                        f'Parameter {param_name} can not be set for '
                        f'compartment {n_comp}. The length of the given '
                        'parameter does not agree with the number of circuits '
                        'in the compartment.')
                else:
                    comp_param = np.asarray(comp_param)

                for neuron_params in self.parameter_space[param_name]:
                    neuron_params.value[n_comp] = comp_param

    @staticmethod
    def _get_initial_config(coord: halco.LogicalNeuronOnDLS,
                            param_name: str) -> McCircuitParameters:
        '''
        Extract the initial configuration from the simulator state.

        :param coord: Coordinate of Logical neuron for which to extract
            the initial configuration.
        :param param_name: Name of parameter for which to extract the initial
            values.
        :returns: Initial configuration saved in the simulator as
            McCircuitParameters.
        '''

        if simulator.state.initial_config is None:
            raise RuntimeError(
                'Can not extract initial configuration since no configuration '
                'is saved in the simulator state.')

        atomic_configs = \
            simulator.state.initial_config.neuron_block.atomic_neurons

        atomic_member_name, member_name = \
            decompose_in_member_names(param_name)
        params_neuron = []
        for compartments in coord.get_placed_compartments().values():
            params_compartment = []
            for an_coord in compartments:
                atomic_config = atomic_configs[an_coord]
                atomic_member = getattr(atomic_config, atomic_member_name)
                value = getattr(atomic_member, member_name)
                # convert CapMem values to integers
                if isinstance(value, hal.CapMemCell.Value):
                    value = value.value()
                params_compartment.append(value)
            params_neuron.append(params_compartment)

        return McCircuitParameters(params_neuron)

    def apply_config(self, coords: List[halco.LogicalNeuronOnDLS]) -> None:
        """
        Apply configuration  initial configuration  saved in the simulator
        state to the parameter space.

        :param coords: List of coordinates for which to look up the initial
            configuration. Needs same order and dimensions as parameter_space.
        """
        if simulator.state.initial_config is None:
            return

        for param_name in self.parameter_space.keys():
            params = []
            for coord in coords:
                params.append(self._get_initial_config(coord, param_name))
            self.parameter_space.update(**{param_name: params})

        # parameters defined for the morphology take precedence
        self._add_morphology_params_to_ps()

        # parameters provided manually by the user have precedence
        self.parameter_space.update(**self._user_provided_parameters)

        # Only use first compartment
        if self.single_active_circuit:
            self._update_all_but_first('leak_i_bias', 0)
            self._update_all_but_first('leak_enable_division', True)
            self._update_all_but_first('leak_enable_multiplication', False)

            self._update_all_but_first('membrane_capacitance_capacitance', 0)
            self._update_all_but_first('threshold_enable', False)

    @classmethod
    def _change_all_but_first_circuit(cls, value: Any, new_value: Any
                                      ) -> McCircuitParameters:
        '''
        Update the values of each compartment and circuit (but the first in
        each compartment) to the new value.

        If `value` is not of type McCircuitParameters, `value` is expanded to
        the shape of the morphology.

        :param value: Value for which to update all but the first circuits.
        :param new_value: New value for all but the first circuits.
        :return: `value` updated such that the first circuits have the same
            values as `value` and all other circuits the value `new_value`.
        '''
        value = deepcopy(value)
        if not isinstance(value, McCircuitParameters):
            tmp_value = []
            for comp in cls.compartments.values():
                tmp_value.append([value] * comp.size)
            value = McCircuitParameters(tmp_value)

        for comp_values in value.value:
            comp_values[1:] = new_value

        return value

    def _update_all_but_first(self, name: str, new_value: Any) -> None:
        '''
        Update all but the value of the first neuron circuit to the new value.

        :param name: Name of parameter in parameter space which will be
            updated.
        :param new_value: Value to which to set the parameters.
        '''

        old_values = self.parameter_space[name]

        if old_values.shape is None:
            new_values = self._change_all_but_first_circuit(
                old_values.base_value, new_value)
            self.parameter_space.update(**{name: new_values})
        else:
            # iterate over different neurons in parameter space
            new_values = []
            for old_value in old_values:
                new_values.append(self._change_all_but_first_circuit(
                    old_value, new_value))
            self.parameter_space.update(**{name: new_values})

    def add_to_chip(self, cell_ids: List, config: lola.Chip) -> None:
        """
        Add configuration of each neuron in the parameter space to the give
        chip object.


        :param cell_ids: Cell IDs for each neuron in the parameter space of
            this celltype object.
        :param chip: Lola chip object which is altered.
        """
        for cell_in_pop, cell_id in enumerate(cell_ids):
            logical_neuron = self.get_logical_neuron(cell_in_pop)
            logical_coord = \
                simulator.state.neuron_placement.id2logicalneuron(cell_id)
            placed_compartments = logical_coord.get_placed_compartments()

            for comp_id, atomic_neurons in logical_neuron.collapse_neuron()\
                    .items():
                for atomic_coord, atomic_config in zip(
                        placed_compartments[comp_id], atomic_neurons):
                    atomic_config.event_routing.analog_output = \
                        atomic_config.EventRouting.AnalogOutputMode.normal
                    config.neuron_block.atomic_neurons[atomic_coord] = \
                        atomic_config


McNeuronBase.translations = McNeuronBase._create_translation()  # pylint: disable=protected-access
