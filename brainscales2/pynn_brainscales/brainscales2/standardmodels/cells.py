# pylint: disable=too-many-lines
import numbers
import copy
from typing import List, Dict, ClassVar, Final, Optional, Union, Callable, \
    Tuple
import numpy as np

from pyNN.parameters import ArrayParameter, ParameterSpace
from pyNN.standardmodels import cells, build_translations
from pyNN.common import Population
from pynn_brainscales.brainscales2 import simulator, plasticity_rules
from pynn_brainscales.brainscales2.helper import (
    get_values_of_atomic_neuron,
    decompose_in_member_names,
)
from pynn_brainscales.brainscales2.standardmodels.cells_base import (
    StandardCellType,
    NeuronCellType,
    ExternalNeuron,
)
from dlens_vx_v3 import lola, hal, halco, sta, hxcomm
import pygrenade_vx.network.abstract as grenade
import pygrenade_vx.network as grenade_network
import pygrenade_vx.common as grenade_vx_common
import pygrenade_common as grenade_common


class HXNeuron(NeuronCellType):
    """
    One to one representation of subset of parameter space of a
    lola.AtomicNeuron. Parameter hierarchy is flattened. Defaults to "silent"
    neuron.

    :param parameters: Mapping of parameters and corresponding values, e.g.
                       dict. Either 1-dimensional or population size
                       dimensions. Default values are overwritten for specified
                       parameters.
    """

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]
    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = False
    injectable: Final[bool] = True

    # the actual unit of `v` is `haldls::vx::CapMemCell::Value`
    # [0–1022]; 1023 means off,
    # but only units included in the `quantity` package are accepted
    units: Final[Dict[str, str]] = {"v": "dimensionless",
                                    "exc_synin": "dimensionless",
                                    "inh_synin": "dimensionless",
                                    "adaptation": "dimensionless",
                                    **NeuronCellType.units}
    # manual list of all parameters which should not be exposed
    _NOT_CONFIGURABLE: Final[List[str]] = [
        "event_routing_analog_output",
        "event_routing_enable_digital",
        "leak_reset_i_bias_source_follower",
        "readout_enable_amplifier",
        "readout_source",
        "readout_enable_buffered_access",
        "readout_i_bias"]

    _hw_entity_setters: ClassVar[Dict[str, Callable]]

    # needed to restore after chip config was applied
    _user_provided_parameters: Optional[Dict[str, Union[int, bool]]]

    # HXNeuron consits of a single compartment with a single circuit
    # pylint: disable-next=invalid-name
    logical_compartments: Final[halco.LogicalNeuronCompartments] = \
        halco.LogicalNeuronCompartments(
            {halco.CompartmentOnLogicalNeuron():
             [halco.AtomicNeuronOnLogicalNeuron()]})

    def __init__(self, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        super().__init__(**parameters)
        parameters.pop("plasticity_rule", None)
        self._user_provided_parameters = parameters
        self._initial_config_applied = False

    def validate_parameter_space(self):
        """
        Raise if the parameter space is not yet valid.

        The parameter space is only valid after the initial
        configuration has been applied. This is only the
        case after mapping.
        """
        if simulator.state.initial_config is not None \
                and not self._initial_config_applied:
            raise RuntimeError(
                "Initial config not yet applied. Parameters are not yet "
                "set (and will be overwritten during mapping). Perform "
                "`pynn.run(None, command=pynn.RunCommand.PREPARE)` to apply "
                "the initial config or run an experiment.")

    @classmethod
    def get_default_values(cls) -> dict:
        """Get the default values."""

        return {
            **NeuronCellType.default_parameters,
            **get_values_of_atomic_neuron(lola.AtomicNeuron(),
                                          cls._NOT_CONFIGURABLE)}

    @classmethod
    def _create_translation(cls) -> dict:
        default_values = cls.get_default_values()
        translation = []
        for key in default_values:
            translation.append((key, key))

        return build_translations(*translation)

    def can_record(self, variable: str, location=None) -> bool:
        del location  # for BSS-2 observables do not depend on the location
        return variable in self.recordable

    @classmethod
    def _generate_hw_entity_setters(cls) -> None:
        """
        Builds setters for creation of Lola Neuron.
        """

        cls._hw_entity_setters = {}

        for param in get_values_of_atomic_neuron(
                lola.AtomicNeuron(), cls._NOT_CONFIGURABLE):
            member, attr = decompose_in_member_names(param)

            def generate_setter(member, attr, param):
                if param == "readout_source":
                    def setter(neuron, value):
                        # enable to hand over integers in value
                        # pyNN internally stores all numerical values as float,
                        # we need to get back the int to cast back to an enum.
                        if isinstance(value, numbers.Real):
                            value = int(value)
                        # set initial values
                        real_member = getattr(neuron, member)
                        setattr(real_member, attr,
                                hal.NeuronConfig.ReadoutSource(value))
                        setattr(neuron, member, real_member)
                else:
                    def setter(neuron, value):
                        # enable to hand over integers in value
                        # pyNN internally stores all numerical values as float,
                        # we need to get back the int to cast back to an enum.
                        if isinstance(value, numbers.Real):
                            value = int(value)
                        # set initial values
                        real_member = getattr(neuron, member)
                        # PyNN uses lazyarrays for value storage; need to
                        # restore original type
                        val = type(getattr(real_member, attr))(value)
                        setattr(real_member, attr, val)
                        setattr(neuron, member, real_member)
                return setter
            cls._hw_entity_setters[param] = \
                generate_setter(member, attr, param)

    @classmethod
    def create_hw_entity(cls, pynn_parameters: dict) -> lola.AtomicNeuron:
        """
        Builds a Lola Neuron with the values from the dict 'pynn_parameters'.
        """

        neuron = lola.AtomicNeuron()
        for param in pynn_parameters:
            if param in cls._hw_entity_setters:
                cls._hw_entity_setters[param](
                    neuron, pynn_parameters[param])

        return neuron

    def apply_config(self, logical_coords: List[halco.LogicalNeuronOnDLS]):
        """
        Extract and apply config according to provided chip object

        :param coords: List of coordinates to look up coco. Needs
                       same order and dimensions as parameter_space.
        """
        if simulator.state.initial_config is None \
                or self._initial_config_applied:
            # no coco provided or already applied -> skip
            return

        # HXneurons consist of single compartments with single circuits
        assert np.all([len(coord.get_atomic_neurons()) == 1 for coord
                       in logical_coords])
        coords = [coord.get_atomic_neurons()[0] for coord in logical_coords]

        param_per_neuron: List[Dict[str, Union[int, bool]]]
        param_per_neuron = []
        param_dict: Dict[str, List[Union[int, bool]]]
        param_dict = {}
        for coord in coords:
            try:
                atomic_neuron = simulator.state.initial_config\
                    .neuron_block.atomic_neurons[coord]
            except KeyError as err:
                raise KeyError(f"No coco entry for {coord}") from err
            param_per_neuron.append(get_values_of_atomic_neuron(
                atomic_neuron, self._NOT_CONFIGURABLE))

        # "fuse" entries of individual parameters to one large dict of arrays
        for k in param_per_neuron[0].keys():
            param_dict[k] = \
                tuple(coco_entry[k] for coco_entry in param_per_neuron)

        self.parameter_space.update(**param_dict)
        # parameters provided manually by the user have precedence -> overwrite
        self.parameter_space.update(**self._user_provided_parameters)
        self._initial_config_applied = True

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        compartment_receptors = [{
            grenade_common.ReceptorOnCompartment(0):
            grenade_network.Receptor.Type.excitatory,
            grenade_common.ReceptorOnCompartment(1):
            grenade_network.Receptor.Type.inhibitory
        }]
        compartment = grenade.UncalibratedNeuron.Compartment(
            grenade.UncalibratedNeuron.Compartment
            .SpikeMaster(0),
            compartment_receptors)
        compartments = {
            grenade_common.CompartmentOnNeuron():
            compartment}

        cell = grenade.UncalibratedNeuron(
            compartments=compartments,
            shape=halco.LogicalNeuronCompartments({
                halco.CompartmentOnLogicalNeuron():
                [halco.AtomicNeuronOnLogicalNeuron()]}))

        shape = grenade_common.CuboidMultiIndexSequence(
            [len(population)],
            [grenade_common.CellOnPopulationDimensionUnit()])

        cell_num_neuron_circuits = {
            grenade_common.CompartmentOnNeuron(): 1}

        parameter_space = grenade.UncalibratedNeuron.ParameterSpace(
            len(population), cell_num_neuron_circuits)

        return grenade_common.Population(
            cell=cell,
            shape=shape,
            parameter_space=parameter_space,
            time_domain=grenade_common.TimeDomainOnTopology())

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        parameterization = grenade.UncalibratedNeuron.ParameterSpace\
            .Parameterization()

        coords = grenade.reverse_mapping.get_locally_placed_neuron_coordinates(
            population.grenade_descriptor, experiment.mapped_topology)
        population.celltype.apply_config(coords)

        parameterization.configs = [
            {grenade_common.CompartmentOnNeuron():
             [HXNeuron.create_hw_entity({
                 key: values[i] for key, values
                 in population.celltype.parameter_space.items()})]}
            for i in range(len(population))]

        for i in range(len(population)):
            # analog output is needed for spiking -> enable globally
            # TODO: move to reasonable place
            parameterization.configs[i][grenade_common.CompartmentOnNeuron()][
                0].event_routing.analog_output = \
                lola.AtomicNeuron.EventRouting.AnalogOutputMode.normal

        # TODO: make more efficient with less duplicate entries
        parameterization.base_configs = [
            (list(range(len(population))),
             simulator.state.get_base_configuration())]

        for i in population.recorder.recorded["v"]:
            parameterization.configs[population.id_to_index(i.cell_id)][
                grenade_common.CompartmentOnNeuron()][
                0].readout.source = lola.AtomicNeuron.Readout.Source.membrane
        for i in population.recorder.recorded["exc_synin"]:
            parameterization.configs[population.id_to_index(i.cell_id)][
                grenade_common.CompartmentOnNeuron()][
                0].readout.source = lola.AtomicNeuron.Readout.Source.exc_synin
        for i in population.recorder.recorded["inh_synin"]:
            parameterization.configs[population.id_to_index(i.cell_id)][
                grenade_common.CompartmentOnNeuron()][
                0].readout.source = lola.AtomicNeuron.Readout.Source.inh_synin
        for i in population.recorder.recorded["adaptation"]:
            parameterization.configs[population.id_to_index(i.cell_id)][
                grenade_common.CompartmentOnNeuron()][
                0].readout.source = lola.AtomicNeuron.Readout.Source.adaptation

        return {1: parameterization}


HXNeuron.default_parameters = HXNeuron.get_default_values()
HXNeuron.translations = HXNeuron._create_translation()  # pylint: disable=protected-access
HXNeuron._generate_hw_entity_setters()  # pylint: disable=protected-access


class CalibHXNeuronCuba(NeuronCellType):
    """
    HX Neuron with automated calibration. Cell parameters correspond to
    parameters for Calix spiking calibration.

    Uses current-based synapses.
    """
    def __init__(
            self,
            plasticity_rule: Optional[plasticity_rules.PlasticityRule] = None,
            *,
            v_rest: Optional[Union[int, List, np.ndarray]] = None,
            v_reset: Optional[Union[int, List, np.ndarray]] = None,
            v_thresh: Optional[Union[int, List, np.ndarray]] = None,
            tau_m: Optional[Union[int, List, np.ndarray]] = None,
            tau_syn_E: Optional[Union[int, List, np.ndarray]] = None,
            tau_syn_I: Optional[Union[int, List, np.ndarray]] = None,
            cm: Optional[Union[int, List, np.ndarray]] = None,
            tau_refrac: Optional[Union[int, List, np.ndarray]] = None,
            i_synin_gm_E: Optional[Union[int, List, np.ndarray]] = None,
            i_synin_gm_I: Optional[Union[int, List, np.ndarray]] = None,
            synapse_dac_bias: Optional[Union[int, List, np.ndarray]] = None,
            **parameters):
        """
        Set initial neuron parameters.

        All parameters except for the plasticity rule have to be either
        1-dimensional or population-size dimensional.

        Additional keyword arguments are also saved in the parameter space of
        the neuron but do not influence the calibration.

        :param plasticity_rule: Plasticity rule which is evaluated periodically
            during the experiment to change the parameters of the neuron.
        :param v_rest: Resting potential. Value range [50, 160].
        :param v_reset: Reset potential. Value range [50, 160].
        :param v_thresh: Threshold potential. Value range [50, 220].
        :param tau_m: Membrane time constant. Value range [0.5, 60]us.
        :param tau_syn_E: Excitatory synaptic input time constant. Value range
            [0.3, 30]us.
        :param tau_syn_I: Inhibitory synaptic input time constant. Value range
            [0.3, 30]us.
        :param cm: Membrane capacitance. Value range [0, 63].
        :param tau_refrac: Refractory time. Value range [.04, 32]us.
        :param i_synin_gm_E: Excitatory synaptic input strength bias current.
            Scales the strength of excitatory weights. Technical parameter
            which needs to be same for all populations. Value range [30, 800].
        :param i_synin_gm_I: Inhibitory synaptic input strength bias current.
            Scales the strength of excitatory weights. Technical parameter
            which needs to be same for all populations. Value range [30, 800].
        :param synapse_dac_bias: Synapse DAC bias current. Technical parameter
            which needs to be same for all populations Can be lowered in order
            to reduce the amplitude of a spike at the input of the synaptic
            input OTA. This can be useful to avoid saturation when using larger
            synaptic time constants. Value range [30, 1022].
        """
        super().__init__(plasticity_rule=plasticity_rule,
                         v_rest=v_rest,
                         v_reset=v_reset,
                         v_thresh=v_thresh,
                         tau_m=tau_m,
                         tau_syn_E=tau_syn_E,
                         tau_syn_I=tau_syn_I,
                         cm=cm,
                         tau_refrac=tau_refrac,
                         i_synin_gm_E=i_synin_gm_E,
                         i_synin_gm_I=i_synin_gm_I,
                         synapse_dac_bias=synapse_dac_bias,
                         **parameters)
        self._calib_target: Optional[ParameterSpace] = None
        self._calib_hwparams: Optional[List[lola.AtomicNeuron]] = None
        self._actual_hwparams: Optional[List[lola.AtomicNeuron]] = None

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]

    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = False

    default_parameters = {
        'v_rest': 80.,
        'v_reset': 70.,
        'v_thresh': 125.,
        'tau_m': 10.,
        'tau_syn_E': 10.,
        'tau_syn_I': 10.,
        'cm': 63,
        'tau_refrac': 2.,
        'i_synin_gm_E': 500,
        'i_synin_gm_I': 500,
        'synapse_dac_bias': 600,
        **NeuronCellType.default_parameters}

    units = {
        'v_rest': 'dimensionless',
        'v_reset': 'dimensionless',
        'v_thresh': 'dimensionless',
        'tau_m': 'us',
        'tau_syn_E': 'us',
        'tau_syn_I': 'us',
        'cm': 'dimensionless',
        'tau_refrac': 'us',
        'i_synin_gm_E': 'dimensionless',
        'i_synin_gm_I': 'dimensionless',
        'synapse_dac_bias': 'dimensionless',
        "v": "dimensionless",
        "exc_synin": "dimensionless",
        "inh_synin": "dimensionless",
        "adaptation": "dimensionless",
        **NeuronCellType.units
    }

    translations = {**build_translations(
        ('v_rest', 'v_rest'),
        ('v_reset', 'v_reset'),
        ('v_thresh', 'v_thresh'),
        ('tau_m', 'tau_m'),
        ('tau_syn_E', 'tau_syn_E'),
        ('tau_syn_I', 'tau_syn_I'),
        ('cm', 'cm'),
        ('tau_refrac', 'tau_refrac'),
        ('i_synin_gm_E', 'i_synin_gm_E'),
        ('i_synin_gm_I', 'i_synin_gm_I'),
        ('synapse_dac_bias', 'synapse_dac_bias'),
        ('v', 'v'),
        ('exc_synin', 'exc_synin'),
        ('inh_synin', 'inh_synin'),
        ('adaptation', 'adaptation')),
        **NeuronCellType.translations}

    # HXNeuron consists of a single compartment with a single circuit
    # pylint: disable-next=invalid-name
    logical_compartments: Final[halco.LogicalNeuronCompartments] = \
        halco.LogicalNeuronCompartments(
            {halco.CompartmentOnLogicalNeuron():
             [halco.AtomicNeuronOnLogicalNeuron()]})

    def can_record(self, variable: str, location=None) -> bool:
        del location  # for BSS-2 observables do not depend on the location
        return variable in self.recordable

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        compartment_receptors = [{
            grenade_common.ReceptorOnCompartment(0):
            grenade_network.Receptor.Type.excitatory,
            grenade_common.ReceptorOnCompartment(1):
            grenade_network.Receptor.Type.inhibitory
        }]
        compartment = grenade.CalibratedNeuron.Compartment(
            grenade.CalibratedNeuron.Compartment
            .SpikeMaster(0),
            compartment_receptors)
        compartments = {
            grenade_common.CompartmentOnNeuron():
            compartment}

        cell = grenade.CalibratedNeuron(
            compartments=compartments,
            shape=halco.LogicalNeuronCompartments({
                halco.CompartmentOnLogicalNeuron():
                [halco.AtomicNeuronOnLogicalNeuron()]}))

        shape = grenade_common.CuboidMultiIndexSequence(
            [len(population)],
            [grenade_common.CellOnPopulationDimensionUnit()])

        parameter_space = grenade.CalibratedNeuron.ParameterSpace(
            CalibHXNeuronCuba.generate_calibration_targets(population))

        return grenade_common.Population(
            cell=cell,
            shape=shape,
            parameter_space=parameter_space,
            time_domain=grenade_common.TimeDomainOnTopology())

    @staticmethod
    def generate_calibration_targets(population: Population) \
            -> List[Dict[grenade_common.CompartmentOnNeuron,
                         List[grenade.CalibratedNeuron
                              .ParameterSpace.CalibrationTarget]]]:
        population.celltype._calib_target = copy.deepcopy(  # pylint: disable=protected-access
            population.celltype.parameter_space)
        calibration_targets = []
        for parameters in population.celltype.parameter_space:
            calibration_target = grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget()
            calibration_target.synaptic_input_excitatory = \
                grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CubaSynapticInput()
            calibration_target.synaptic_input_inhibitory = \
                grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CubaSynapticInput()

            calibration_target.membrane_capacitance = lola.AtomicNeuron\
                .MembraneCapacitance.CapacitorSize(parameters["cm"])
            calibration_target.v_leak = int(parameters["v_rest"])
            calibration_target.tau_membrane = float(parameters["tau_m"]) * 1e-6

            calibration_target.v_threshold = int(parameters["v_thresh"])
            calibration_target.v_reset = int(parameters["v_reset"])

            calibration_target.refractory_period.refractory_time = \
                float(parameters["tau_refrac"]) * 1e-6

            calibration_target.synaptic_input_excitatory.i_synin_gm = \
                int(parameters["i_synin_gm_E"])
            calibration_target.synaptic_input_inhibitory.i_synin_gm = \
                int(parameters["i_synin_gm_I"])

            calibration_target.synaptic_input_excitatory.tau_syn = \
                float(parameters["tau_syn_E"]) * 1e-6
            calibration_target.synaptic_input_inhibitory.tau_syn = \
                float(parameters["tau_syn_I"]) * 1e-6

            calibration_target.synaptic_input_excitatory.synapse_dac_bias = \
                int(parameters["synapse_dac_bias"])
            calibration_target.synaptic_input_inhibitory.synapse_dac_bias = \
                int(parameters["synapse_dac_bias"])

            calibration_targets.append(
                {grenade_common.CompartmentOnNeuron(): [calibration_target]})
        return calibration_targets

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        logical_neuron_configs = grenade.reverse_mapping\
            .get_calibrated_neuron_actual_hardware_parameters(
                population.grenade_descriptor,
                experiment.mapped_topology)

        configs = []
        for logical_neuron in logical_neuron_configs:
            configs.append(
                logical_neuron[grenade_common.CompartmentOnNeuron()][0])

        self._calib_hwparams = copy.deepcopy(configs)
        self._actual_hwparams = configs

        # readout source defaulting to membrane
        readout_sources = [{grenade_common.CompartmentOnNeuron():
                            [lola.AtomicNeuron.Readout.Source.membrane]}
                           for i in range(len(population))]
        for i in population.recorder.recorded["exc_synin"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.exc_synin]}
        for i in population.recorder.recorded["inh_synin"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.inh_synin]}
        for i in population.recorder.recorded["adaptation"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.adaptation]}

        return {1: grenade.CalibratedNeuron.ParameterSpace
                .Parameterization(
                    CalibHXNeuronCuba.generate_calibration_targets(population),
                    readout_sources)}

    # map between pynn and hardware parameter names. Cannot utilize pyNN
    # translations as it need to be bijective
    param_trans = {
        'v_rest': 'leak',
        'v_reset': 'reset',
        'v_thresh': 'threshold',
        'tau_m': 'tau_mem',
        'tau_syn_E': 'tau_syn',
        'tau_syn_I': 'tau_syn',
        'cm': 'membrane_capacitance',
        'tau_refrac': 'refractory_time',
        'i_synin_gm_E': 'i_synin_gm',
        'i_synin_gm_I': 'i_synin_gm',
        'synapse_dac_bias': 'synapse_dac_bias',
        'v': 'v',
        'exc_synin': 'exc_synin',
        'inh_synin': 'inh_synin',
        'adaptation': 'adaptation',
        **{key: translation["translated_name"]
           for key, translation in NeuronCellType.translations.items()}}

    @property
    def actual_hwparams(self) -> Optional[Tuple[lola.AtomicNeuron]]:
        """
        Hardware parameters used for actual hardware execution, can be
        manually adjusted.

        Only set after pynn.run() call. You do not need to execute the
        experiment but can also use `pynn.run(None, pynn.RunComand.PREPARE)`.
        """
        # cast to tuple prevents overwriting reference with new object
        if self._actual_hwparams is None:
            raise AttributeError("actual_hwparams only available after "
                                 "pynn.run() call. You can als use "
                                 "pynn.run(None, pynn.RunCommand.PREPARE).")
        return tuple(self._actual_hwparams)

    @property
    def calib_hwparams(self) -> Optional[List[lola.AtomicNeuron]]:
        """
        Archive of resulting hardware parameters from last calibration run.

        Only set after pynn.run() call. You do not need to execute the
        experiment but can also use `pynn.run(None, pynn.RunComand.PREPARE)`.
        """
        if self._calib_hwparams is None:
            raise AttributeError("actual_hwparams only available after "
                                 "pynn.run() call. You can als use "
                                 "pynn.run(None, pynn.RunCommand.PREPARE).")
        return self._calib_hwparams

    @property
    def calib_target(self) -> Optional[ParameterSpace]:
        """
        Archive of cell parameters used for last calibration run.

        Only set after pynn.run() call. You do not need to execute the
        experiment but can also use `pynn.run(None, pynn.RunComand.PREPARE)`.
        """
        if self._calib_target is None:
            raise AttributeError("actual_hwparams only available after "
                                 "pynn.run() call. You can als use "
                                 "pynn.run(None, pynn.RunCommand.PREPARE).")
        return self._calib_target


class CalibHXNeuronCoba(CalibHXNeuronCuba):
    """
    HX Neuron with automated calibration. Cell parameters correspond to
    parameters for Calix spiking calibration.

    Uses conductance-based synapses.
    """

    # pylint: disable=too-many-locals
    def __init__(
            self,
            plasticity_rule: Optional[plasticity_rules.PlasticityRule] = None,
            *,
            v_rest: Optional[Union[int, List, np.ndarray]] = None,
            v_reset: Optional[Union[int, List, np.ndarray]] = None,
            v_thresh: Optional[Union[int, List, np.ndarray]] = None,
            tau_m: Optional[Union[int, List, np.ndarray]] = None,
            tau_syn_E: Optional[Union[int, List, np.ndarray]] = None,
            tau_syn_I: Optional[Union[int, List, np.ndarray]] = None,
            cm: Optional[Union[int, List, np.ndarray]] = None,
            tau_refrac: Optional[Union[int, List, np.ndarray]] = None,
            i_synin_gm_E: Optional[Union[int, List, np.ndarray]] = None,
            i_synin_gm_I: Optional[Union[int, List, np.ndarray]] = None,
            e_rev_E: Optional[Union[int, List, np.ndarray]] = None,
            e_rev_I: Optional[Union[int, List, np.ndarray]] = None,
            synapse_dac_bias: Optional[Union[int, List, np.ndarray]] = None,
            **parameters):
        """
        Set initial neuron parameters.

        All parameters except for the plasticity rule have to be either
        1-dimensional or population-size dimensional.

        Additional key-word arguments are also saved in the parameter space of
        the neuron but do not influence the calibration.

        :param plasticity_rule: Plasticity rule which is evaluated periodically
            during the experiment to change the parameters of the neuron.
        :param v_rest: Resting potential. Value range [50, 160].
        :param v_reset: Reset potential. Value range [50, 160].
        :param v_thresh: Threshold potential. Value range [50, 220].
        :param tau_m: Membrane time constant. Value range [0.5, 60]us.
        :param tau_syn_E: Excitatory synaptic input time constant. Value range
            [0.3, 30]us.
        :param tau_syn_I: Inhibitory synaptic input time constant. Value range
            [0.3, 30]us.
        :param cm: Membrane capacitance. Value range [0, 63].
        :param tau_refrac: Refractory time. Value range [.04, 32]us.
        :param i_synin_gm_E: Excitatory synaptic input strength bias current.
            Scales the strength of excitatory weights. Technical parameter
            which needs to be same for all populations. Value range [30, 800].
        :param i_synin_gm_I: Inhibitory synaptic input strength bias current.
            Scales the strength of excitatory weights. Technical parameter
            which needs to be same for all populations. Value range [30, 800].
        :param e_rev_E: Excitatory COBA synaptic input reversal potential. At
            this potential, the synaptic input strength will be zero. Value
            range [60, 160].
        :param e_rev_I: Inhibitory COBA synaptic input reversal potential. At
            this potential, the synaptic input strength will be zero. Value
            range [60, 160].
        :param synapse_dac_bias: Synapse DAC bias current. Technical parameter
            which needs to be same for all populations Can be lowered in order
            to reduce the amplitude of a spike at the input of the synaptic
            input OTA. This can be useful to avoid saturation when using larger
            synaptic time constants. Value range [30, 1022].
        """
        super().__init__(plasticity_rule=plasticity_rule,
                         v_rest=v_rest,
                         v_reset=v_reset,
                         v_thresh=v_thresh,
                         tau_m=tau_m,
                         tau_syn_E=tau_syn_E,
                         tau_syn_I=tau_syn_I,
                         cm=cm,
                         tau_refrac=tau_refrac,
                         i_synin_gm_E=i_synin_gm_E,
                         i_synin_gm_I=i_synin_gm_I,
                         e_rev_E=e_rev_E,
                         e_rev_I=e_rev_I,
                         synapse_dac_bias=synapse_dac_bias,
                         **parameters)

    conductance_based: Final[bool] = True

    default_parameters = {
        'v_rest': 110.,
        'v_reset': 100.,
        'v_thresh': 160.,
        'tau_m': 10.,
        'tau_syn_E': 10.,
        'tau_syn_I': 10.,
        'cm': 63,
        'tau_refrac': 2.,
        'e_rev_E': 300.,
        'e_rev_I': 30.,
        'i_synin_gm_E': 180,
        'i_synin_gm_I': 250,
        'synapse_dac_bias': 400,
        **NeuronCellType.default_parameters}

    units = {
        'v_rest': 'dimensionless',
        'v_reset': 'dimensionless',
        'v_thresh': 'dimensionless',
        'tau_m': 'us',
        'tau_syn_E': 'us',
        'tau_syn_I': 'us',
        'cm': 'dimensionless',
        'refractory_time': 'us',
        'e_rev_E': 'dimensionless',
        'e_rev_I': 'dimensionless',
        'i_synin_gm_E': 'dimensionless',
        'i_synin_gm_I': 'dimensionless',
        'synapse_dac_bias': 'dimensionless',
        "v": "dimensionless",
        "exc_synin": "dimensionless",
        "inh_synin": "dimensionless",
        "adaptation": "dimensionless",
        **NeuronCellType.units
    }

    translations = {**build_translations(
        ('v_rest', 'v_rest'),
        ('v_reset', 'v_reset'),
        ('v_thresh', 'v_thresh'),
        ('tau_m', 'tau_m'),
        ('tau_syn_E', 'tau_syn_E'),
        ('tau_syn_I', 'tau_syn_I'),
        ('cm', 'cm'),
        ('tau_refrac', 'tau_refrac'),
        ('e_rev_E', 'e_rev_E'),
        ('e_rev_I', 'e_rev_I'),
        ('i_synin_gm_E', 'i_synin_gm_E'),
        ('i_synin_gm_I', 'i_synin_gm_I'),
        ('synapse_dac_bias', 'synapse_dac_bias'),
        ('v', 'v'),
        ('exc_synin', 'exc_synin'),
        ('inh_synin', 'inh_synin'),
        ('adaptation', 'adaptation')),
        **NeuronCellType.translations}

    # map between pynn and hardware parameter names. Cannot utilize pyNN
    # translations as it need to be bijective
    param_trans = {
        'v_rest': 'leak',
        'v_reset': 'reset',
        'v_thresh': 'threshold',
        'tau_m': 'tau_mem',
        'tau_syn_E': 'tau_syn',
        'tau_syn_I': 'tau_syn',
        'cm': 'membrane_capacitance',
        'tau_refrac': 'refractory_time',
        'e_rev_E': 'e_coba_reversal',
        'e_rev_I': 'e_coba_reversal',
        'i_synin_gm_E': 'i_synin_gm',
        'i_synin_gm_I': 'i_synin_gm',
        'synapse_dac_bias': 'synapse_dac_bias',
        'v': 'v',
        'exc_synin': 'exc_synin',
        'inh_synin': 'inh_synin',
        'adaptation': 'adaptation',
        **{key: translation["translated_name"]
           for key, translation in NeuronCellType.translations.items()}}

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        compartment_receptors = [{
            grenade_common.ReceptorOnCompartment(0):
            grenade_network.Receptor.Type.excitatory,
            grenade_common.ReceptorOnCompartment(1):
            grenade_network.Receptor.Type.inhibitory
        }]
        compartment = grenade.CalibratedNeuron.Compartment(
            grenade.CalibratedNeuron.Compartment
            .SpikeMaster(0),
            compartment_receptors)
        compartments = {
            grenade_common.CompartmentOnNeuron():
            compartment}

        cell = grenade.CalibratedNeuron(
            compartments=compartments,
            shape=halco.LogicalNeuronCompartments({
                halco.CompartmentOnLogicalNeuron():
                [halco.AtomicNeuronOnLogicalNeuron()]}))

        shape = grenade_common.CuboidMultiIndexSequence(
            [len(population)],
            [grenade_common.CellOnPopulationDimensionUnit()])

        parameter_space = grenade.CalibratedNeuron.ParameterSpace(
            CalibHXNeuronCoba.generate_calibration_targets(population))

        return grenade_common.Population(
            cell=cell,
            shape=shape,
            parameter_space=parameter_space,
            time_domain=grenade_common.TimeDomainOnTopology())

    @staticmethod
    def generate_calibration_targets(population: Population) \
            -> List[Dict[grenade_common.CompartmentOnNeuron,
                         List[grenade.CalibratedNeuron
                              .ParameterSpace.CalibrationTarget]]]:
        # pylint: disable=protected-access
        population.celltype._calib_target = copy.deepcopy(
            population.celltype.parameter_space)
        calibration_targets = []
        for parameters in population.celltype.parameter_space:
            calibration_target = grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget()
            calibration_target.synaptic_input_excitatory = \
                grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CobaSynapticInput()
            calibration_target.synaptic_input_inhibitory = \
                grenade.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CobaSynapticInput()

            calibration_target.membrane_capacitance = lola.AtomicNeuron\
                .MembraneCapacitance.CapacitorSize(parameters["cm"])
            calibration_target.v_leak = int(parameters["v_rest"])
            calibration_target.tau_membrane = float(parameters["tau_m"]) * 1e-6

            calibration_target.v_threshold = int(parameters["v_thresh"])
            calibration_target.v_reset = int(parameters["v_reset"])

            calibration_target.refractory_period.refractory_time = \
                float(parameters["tau_refrac"]) * 1e-6

            calibration_target.synaptic_input_excitatory.i_synin_gm = \
                int(parameters["i_synin_gm_E"])
            calibration_target.synaptic_input_inhibitory.i_synin_gm = \
                int(parameters["i_synin_gm_I"])

            calibration_target.synaptic_input_excitatory.tau_syn = \
                float(parameters["tau_syn_E"]) * 1e-6
            calibration_target.synaptic_input_inhibitory.tau_syn = \
                float(parameters["tau_syn_I"]) * 1e-6

            calibration_target.synaptic_input_excitatory.e_reversal = \
                int(parameters["e_rev_E"])
            calibration_target.synaptic_input_inhibitory.e_reversal = \
                int(parameters["e_rev_I"])

            calibration_target.synaptic_input_excitatory.synapse_dac_bias = \
                int(parameters["synapse_dac_bias"])
            calibration_target.synaptic_input_inhibitory.synapse_dac_bias = \
                int(parameters["synapse_dac_bias"])

            calibration_targets.append(
                {grenade_common.CompartmentOnNeuron(): [calibration_target]})
        return calibration_targets

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        logical_neuron_configs = grenade.reverse_mapping\
            .get_calibrated_neuron_actual_hardware_parameters(
                population.grenade_descriptor,
                experiment.mapped_topology)

        configs = []
        for logical_neuron in logical_neuron_configs:
            configs.append(
                logical_neuron[grenade_common.CompartmentOnNeuron()][0])

        self._calib_hwparams = copy.deepcopy(configs)
        self._actual_hwparams = configs

        # readout source defaulting to membrane
        readout_sources = [{grenade_common.CompartmentOnNeuron():
                            [lola.AtomicNeuron.Readout.Source.membrane]}
                           for i in range(len(population))]
        for i in population.recorder.recorded["exc_synin"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.exc_synin]}
        for i in population.recorder.recorded["inh_synin"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.inh_synin]}
        for i in population.recorder.recorded["adaptation"]:
            readout_sources[population.id_to_index(i.cell_id)] = \
                {grenade_common.CompartmentOnNeuron():
                 [lola.AtomicNeuron.Readout.Source.adaptation]}

        return {1: grenade.CalibratedNeuron.ParameterSpace
                .Parameterization(
                    CalibHXNeuronCoba.generate_calibration_targets(population),
                    readout_sources)}


class SpikeSourcePoissonOnChip(StandardCellType):
    """
    Spike source, generating spikes according to a Poisson process.
    """

    def __init__(self, rate, seed):

        parameters = {"rate": rate, "seed": seed}
        super().__init__(**parameters)

    translations = build_translations(
        ('rate', 'rate'),
        ('seed', 'seed'),
    )

    background_source_clock_freq: ClassVar[float]

    recordable: Final[List[str]] = ['spikes']

    def can_record(self, variable: str, location=None) -> bool:
        del location  # for BSS-2 observables do not depend on the location
        return variable in self.recordable

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        return grenade_common.Population(
            grenade.PoissonSourceNeuron(),
            grenade_common.CuboidMultiIndexSequence(
                [len(population)],
                [grenade_common.CellOnPopulationDimensionUnit()]),
            grenade.PoissonSourceNeuron.ParameterSpace(len(population)),
            grenade_common.TimeDomainOnTopology())

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        # calculate period and rate from rate[Hz]
        assert np.all(population.celltype.parameter_space["rate"]
                      == population.celltype.parameter_space["rate"][0])
        assert np.all(population.celltype.parameter_space["seed"]
                      == population.celltype.parameter_space["seed"][0])
        hwrate = population.celltype.parameter_space["rate"][0] \
            * population.size
        if hwrate > SpikeSourcePoissonOnChip.background_source_clock_freq:
            raise RuntimeError(
                "The chosen Poisson rate can not be realized on"
                " hardware. The product of rate and number of neurons is too"
                " high in this population.")
        rate = hal.BackgroundSpikeSource.Rate(int(round(
            hal.BackgroundSpikeSource.Rate.max * hwrate
            / SpikeSourcePoissonOnChip.background_source_clock_freq)))
        prob = (rate.value() + 1) / hal.BackgroundSpikeSource.Rate.size
        period = hal.BackgroundSpikeSource.Period(
            int(round(SpikeSourcePoissonOnChip
                .background_source_clock_freq / hwrate * prob) - 1))
        # create grenade parameterization
        parameterization = grenade.PoissonSourceNeuron.ParameterSpace\
            .Parameterization(len(population))
        parameterization.period = period
        parameterization.rate = rate
        parameterization.seed = hal.BackgroundSpikeSource.Seed(
            population.celltype.parameter_space["seed"][0])
        return {0: parameterization}


# pylint: disable=no-member,unsubscriptable-object
SpikeSourcePoissonOnChip.background_source_clock_freq = \
    sta.ChipInit(hxcomm.ZeroMockEntry()) \
    .adplls[sta.ChipInit(hxcomm.ZeroMockEntry()).pll_clock_output_block
            .get_clock_output(
                halco.PLLClockOutputOnDLS.phy_ref_clk).select_adpll] \
    .calculate_output_frequency(
        sta.ChipInit(hxcomm.ZeroMockEntry()).pll_clock_output_block
        .get_clock_output(halco.PLLClockOutputOnDLS.phy_ref_clk)
        .select_adpll_output) / 2.  # spl1_clk
# pylint: enable=no-member,unsubscriptable-object


class SpikeSourcePoisson(ExternalNeuron, cells.SpikeSourcePoisson):
    """
    Spike source, generating spikes according to a Poisson process.
    """

    def __init__(self, start, rate, duration):

        parameters = {"start": start,
                      "rate": rate,
                      "duration": duration}
        super().__init__(**parameters)

    translations = build_translations(
        ('start', 'start'),
        ('rate', 'rate'),
        ('duration', 'duration'),
    )

    # Add internal members for the creation of spike trains.
    # 'spike_times' memorizes a fixed Poisson stimulus (for each neuron in the
    # population). 'used_parameters' stores the parameters which were used to
    # generate these spike trains
    _spike_times: Optional[List[np.ndarray]] = None
    _used_parameters: Optional[Dict[str, np.ndarray]] = None

    recordable: Final[List[str]] = ['spikes']

    def can_record(self, variable: str, location=None) -> bool:
        del location  # for BSS-2 observables do not depend on the location
        return variable in self.recordable

    def get_spike_times(self) -> List[np.ndarray]:
        """
        When this function is called for the first time, the spike times for a
        Poisson stimulation are calculated and saved, so that all neurons
        connected to it receive the same stimulation. When a parameter was
        changed (compared to the last calculation of the spike time), the times
        are recalculated.

        :return: (unsorted) spike times for each neuron in the population.
        """

        # generate empty arrays for '_spike_times' and '_used_paramters' if
        # spike times are calculated for the first time
        if self._spike_times is None:
            # Get number of neurons in population from 'start' parameter
            pop_size = self.parameter_space["start"].shape[0]

            self._used_parameters = {"start": np.zeros(pop_size),
                                     "rate": np.zeros(pop_size),
                                     "duration": np.zeros(pop_size)}
            self._spike_times = [np.array([])] * pop_size

        # Recalculate spike times for which at least one parameter changed
        param_space = self.parameter_space
        change = (param_space["start"] != self._used_parameters["start"]) | \
            (param_space["rate"] != self._used_parameters["rate"]) | \
            (param_space["duration"] != self._used_parameters["duration"])

        for neuron_idx in np.where(change)[0]:
            start = self.parameter_space["start"][neuron_idx]
            rate = self.parameter_space["rate"][neuron_idx]
            duration = self.parameter_space["duration"][neuron_idx]

            rate_in_per_ms = rate / 1000  # Convert rate from 1/s in 1/ms
            num_spikes = np.random.poisson(int(duration * rate_in_per_ms))
            self._spike_times[neuron_idx] = start + \
                np.random.rand(num_spikes) * duration

        # Save parameters which were used to generate the spike trains
        self._used_parameters["start"] = self.parameter_space["start"]
        self._used_parameters["rate"] = self.parameter_space["rate"]
        self._used_parameters["duration"] = self.parameter_space["duration"]

        return self._spike_times

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        return SpikeSourceArray.generate_vertex(population)

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        spiketimes = population.celltype.get_spike_times()
        filtered_spiketimes = []
        for spiketimes_neuron in spiketimes:
            filtered_spiketimes_neuron = spiketimes_neuron[
                np.logical_and(
                    snippet_begin_time <= spiketimes_neuron,
                    spiketimes_neuron < snippet_end_time)] \
                - snippet_begin_time
            filtered_spiketimes.append([
                grenade_vx_common.Time(int(time))
                for time in filtered_spiketimes_neuron
                * 1000.
                * grenade_vx_common.Time.fpga_clock_cycles_per_us.value()])

        return {0: grenade.ExternalSourceNeuron.Dynamics(
                [filtered_spiketimes])}


class SpikeSourceArray(ExternalNeuron, cells.SpikeSourceArray):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    # FIXME: workaround, see https://github.com/NeuralEnsemble/PyNN/issues/709
    default_parameters = {'spike_times': ArrayParameter([])}

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )

    recordable: Final[List[str]] = ['spikes']

    def can_record(self, variable: str, location=None) -> bool:
        del location  # for BSS-2 observables do not depend on the location
        return variable in self.recordable

    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        return grenade_common.Population(
            grenade.ExternalSourceNeuron(),
            grenade_common.CuboidMultiIndexSequence(
                [len(population)],
                [grenade_common.CellOnPopulationDimensionUnit()]),
            grenade.ExternalSourceNeuron.ParameterSpace(len(population)),
            grenade_common.TimeDomainOnTopology())

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        spiketimes = population.celltype.parameter_space["spike_times"]
        filtered_spiketimes = []
        for spiketimes_neuron in spiketimes:
            filtered_spiketimes_neuron = spiketimes_neuron.value[
                np.logical_and(
                    snippet_begin_time <= spiketimes_neuron.value,
                    spiketimes_neuron.value < snippet_end_time)] \
                - snippet_begin_time
            filtered_spiketimes.append([
                grenade_vx_common.Time(int(time))
                for time in filtered_spiketimes_neuron
                * 1000.
                * grenade_vx_common.Time.fpga_clock_cycles_per_us.value()])

        return {0: grenade.ExternalSourceNeuron.Dynamics(
                [filtered_spiketimes])}


class SpikeIOCell(ExternalNeuron, StandardCellType):
    """
    Base class for OffChipSource and in the future OffChipSink
    """

    default_parameters = {
        "enable_internal_loopback": True,
        "data_rate_scaler": 1,
        "label": None,
    }
    recordable: Final[List[str]] = ["spikes"]

    def can_record(self, variable: str, location=None) -> bool:
        del location
        return variable in self.recordable

    def __init__(
            self,
            *,
            enable_internal_loopback: bool = True,
            data_rate_scaler: int = 1,
            label: List[int] = None,
    ):
        if label is None or len(label) == 0:
            raise ValueError("OffChipSource: label must be provided "
                             "(one per neuron).")
        super().__init__(
            enable_internal_loopback=enable_internal_loopback,
            data_rate_scaler=data_rate_scaler,
            label=label,
        )

    @staticmethod
    def add_to_input_generator(*args, **kwargs):
        pass


class OffChipSource(SpikeIOCell):
    """
    SpikeIO Input (RX) population
    """
    @staticmethod
    def generate_vertex(population: Population) \
            -> grenade_common.Population:
        return grenade_common.Population(
            grenade.SpikeIOSourceNeuron(),
            grenade_common.CuboidMultiIndexSequence(
                [len(population)],
                [grenade_common.CellOnPopulationDimensionUnit()]),
            grenade.SpikeIOSourceNeuron.ParameterSpace(len(population)),
            grenade_common.TimeDomainOnTopology())

    def generate_input_data(
            self,
            population: Population,
            experiment: grenade.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float) \
            -> Dict[int, grenade_common.PortData]:
        param_space = population.celltype.parameter_space

        for k in ("enable_internal_loopback", "data_rate_scaler"):
            values = np.asarray(param_space[k])
            if not np.all(values == values[0]):
                raise ValueError(
                    f"OffChipSource parameter '{k}' must be uniform across the"
                    f"population, but got values {values!r}."
                )
        enable_internal_loopback = (
            bool(param_space["enable_internal_loopback"][0]))
        data_rate_scaler = int(param_space["data_rate_scaler"][0])

        label = param_space["label"]
        if label is None or len(label) == 0:
            raise ValueError("OffChipSource: label must be provided (one per "
                             "neuron).")
        # left out of uniformity check since now ids are nonuniform
        if len(label) != len(population.all_cells):
            raise ValueError("Internal PyNN Error: Parameter space size "
                             "mismatch")
        # check for double ids
        label_list = [int(x) for x in label]
        if len(label_list) != len(set(label_list)):
            raise ValueError("OffChipSource: label must be unique"
                             " within the population.")

        parameterization = grenade.SpikeIOSourceNeuron.Parameterization()
        parameterization.config.enable_internal_loopback = \
            enable_internal_loopback
        parameterization.config.data_rate_scaler = data_rate_scaler
        # assigns per neuron sensor ids
        parameterization.labels = label_list

        return {0: parameterization}
