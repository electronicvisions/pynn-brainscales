import inspect
import numbers
import copy
import numpy as np
from typing import List, Dict, ClassVar, Final, Optional, Union, Callable, Set

from pyNN.parameters import ArrayParameter, ParameterSpace
from pyNN.standardmodels import cells, build_translations, StandardCellType
from pyNN.common import Population
from pynn_brainscales.brainscales2 import simulator, plasticity_rules
from pynn_brainscales.brainscales2.recording import Recorder
from pynn_brainscales.brainscales2.helper import get_values_of_atomic_neuron, \
    decompose_in_member_names
from dlens_vx_v3 import lola, hal, halco, sta
import pygrenade_vx.network.placed_logical as grenade
from quantities.quantity import Quantity

from pynn_brainscales.brainscales2.standardmodels.cells_base import \
    NetworkAddableCell


class HXNeuron(StandardCellType, NetworkAddableCell):
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
                                    "adaptation": "dimensionless"}
    # manual list of all parameters which should not be exposed
    _not_configurable: Final[List[str]] = [
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
    logical_compartments: Final[halco.LogicalNeuronCompartments] = \
        halco.LogicalNeuronCompartments(
            {halco.CompartmentOnLogicalNeuron():
             [halco.AtomicNeuronOnLogicalNeuron()]})

    def __init__(self, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        self._user_provided_parameters = parameters
        super().__init__(**parameters)

    @classmethod
    def get_default_values(cls) -> dict:
        """Get the default values of a LoLa Neuron."""

        return get_values_of_atomic_neuron(lola.AtomicNeuron(),
                                           cls._not_configurable)

    @classmethod
    def _create_translation(cls) -> dict:
        default_values = cls.get_default_values()
        translation = []
        for key in default_values:
            translation.append((key, key))

        return build_translations(*translation)

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    @classmethod
    def _generate_hw_entity_setters(cls) -> None:
        """
        Builds setters for creation of Lola Neuron.
        """

        cls._hw_entity_setters = dict()

        for param in cls.get_default_values():
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
        if simulator.state.initial_config is None:
            # no coco provided -> skip
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
                atomic_neuron = simulator.state.initial_config.neuron_block.\
                    atomic_neurons[coord]
            except KeyError:
                raise KeyError(f"No coco entry for {coord}")
            param_per_neuron.append(get_values_of_atomic_neuron(
                atomic_neuron, self._not_configurable))

        # "fuse" entries of individual parameters to one large dict of arrays
        for k in param_per_neuron[0].keys():
            param_dict[k] = \
                tuple(coco_entry[k] for coco_entry in param_per_neuron)

        self.parameter_space.update(**param_dict)
        # parameters provided manually by the user have precedence -> overwrite
        self.parameter_space.update(**self._user_provided_parameters)

    @staticmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        # pyNN is more performant when operating on integer cell ids
        pop_cells_int = np.asarray(population.all_cells, dtype=int)

        # get neuron coordinates
        coords: List[halco.LogicalNeuronOnDLS] = \
            simulator.state.neuron_placement.id2logicalneuron(
                population.all_cells)  # pop_cells_int is slower here
        # create receptors
        receptors = set([
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.excitatory),
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.inhibitory),
        ])
        # get recorder configuration
        enable_record_spikes = np.zeros((len(coords)), dtype=bool)
        if "spikes" in population.recorder.recorded:
            enable_record_spikes = np.isin(
                pop_cells_int,
                list(population.recorder.recorded["spikes"]))
        # create neurons
        neurons: List[grenade.Population.Neuron] = [
            grenade.Population.Neuron(
                coord,
                {halco.CompartmentOnLogicalNeuron():
                 grenade.Population.Neuron.Compartment(
                     grenade.Population
                     .Neuron.Compartment.SpikeMaster(
                         0, enable_record_spikes[i]), [receptors])})
            for i, coord in enumerate(coords)
        ]
        # create grenade population
        gpopulation = grenade.Population(neurons)
        # add to builder
        descriptor = builder.add(gpopulation)

        # Terminate early if we don't record
        if simulator.state.madc_recorder is None:
            return descriptor

        # MADC enabled, but nothing to record in this population
        if not (set(population.recorder.recorded)
                & set(Recorder.madc_variables)):
            return descriptor

        # Find the recorded cell
        readout_cell_idxs = np.where(pop_cells_int
                                     == int(simulator.state.
                                            madc_recorder.cell_id))[0]

        # add MADC recording
        assert len(readout_cell_idxs) == 1, "Number of readout cells != 1."
        madc_recording = grenade.MADCRecording()
        madc_recording.population = descriptor
        madc_recording.source = simulator.state.madc_recorder.readout_source
        madc_recording.neuron_on_population = readout_cell_idxs[0]
        madc_recording.compartment_on_neuron = \
            halco.CompartmentOnLogicalNeuron()
        madc_recording.atomic_neuron_on_compartment = 0
        builder.add(madc_recording)

        return descriptor

    @staticmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):
        pass

    # TODO circular import -> can not import ID for type hint of conn_indices
    def add_to_chip(self, cell_ids: List, config: lola.Chip):
        """
        Add configuration of each neuron in the parameter space to the give
        chip object.


        :param cell_ids: Cell IDs for each neuron in the parameter space of
            this celltype object.
        :param chip: Lola chip object which is altered.
        """
        for cell_id, parameters in zip(cell_ids, self.parameter_space):
            atomic_neuron = self.create_hw_entity(parameters)

            # anlog output is needed for spiking -> enable globally
            atomic_neuron.event_routing.analog_output = \
                atomic_neuron.EventRouting.AnalogOutputMode.normal

            # HXNeurons consist of single compartments with single circuits
            logical_coord = \
                simulator.state.neuron_placement.id2logicalneuron(cell_id)
            assert len(logical_coord.get_atomic_neurons()) == 1
            coord = logical_coord.get_atomic_neurons()[0]

            config.neuron_block.atomic_neurons[coord] = atomic_neuron


HXNeuron.default_parameters = HXNeuron.get_default_values()
HXNeuron.translations = HXNeuron._create_translation()  # pylint: disable=protected-access
HXNeuron._generate_hw_entity_setters()  # pylint: disable=protected-access


class PlasticHXNeuron(HXNeuron, plasticity_rules.PlasticityRuleHandle):
    """
    HXNeuron with plasticity rule handle.

    Currently this includes setting the readout source of each neuron
    to be available to the plasticity rule.
    """

    ReadoutSource = lola.AtomicNeuron.Readout.Source

    translations = {
        **HXNeuron.translations, **build_translations(
            ("plasticity_rule_readout_source",
             "plasticity_rule_readout_source"),
            ("plasticity_rule_enable_readout_source",
             "plasticity_rule_enable_readout_source"),
        )}

    def __init__(
            self,
            plasticity_rule_enable_readout_source: bool = False,
            plasticity_rule_readout_source: lola.AtomicNeuron.Readout.Source =
            lola.AtomicNeuron.Readout.Source.membrane,
            plasticity_rule: Optional[plasticity_rules.PlasticityRule] = None,
            **parameters):
        plasticity_rules.PlasticityRuleHandle.__init__(
            self, plasticity_rule=plasticity_rule)
        HXNeuron.__init__(self, **{
            **parameters,
            "plasticity_rule_readout_source": float(
                plasticity_rule_readout_source),
            "plasticity_rule_enable_readout_source":
                plasticity_rule_enable_readout_source})

    @classmethod
    def to_plasticity_rule_population_handle(cls, population: Population) \
            -> grenade.PlasticityRule.PopulationHandle:
        handle = plasticity_rules.PlasticityRuleHandle\
            .to_plasticity_rule_population_handle(population)
        readout_source = population.get(
            "plasticity_rule_readout_source", simplify=False)
        enable_readout_source = population.get(
            "plasticity_rule_enable_readout_source", simplify=False)
        handle.neuron_readout_sources = [
            {halco.CompartmentOnLogicalNeuron(): [
                cls.ReadoutSource(int(readout_source[i])) if
                enable_readout_source[i] else None]}
            for i in range(len(readout_source))
        ]
        return handle

    @classmethod
    def get_default_values(cls) -> dict:
        """Get the default values of a LoLa Neuron."""

        return {**HXNeuron.get_default_values(),
                "plasticity_rule_readout_source": float(
                    cls.ReadoutSource.membrane),
                "plasticity_rule_enable_readout_source": False}


PlasticHXNeuron.default_parameters = PlasticHXNeuron.get_default_values()


class CalibHXNeuronCuba(StandardCellType, NetworkAddableCell):
    """
    HX Neuron with automated calibration. Cell parameters correspond to
    parameters for Calix spiking calibration.

    :param parameters: Mapping of parameters and corresponding values, i.e.
                       dict. Either 1-dimensional or population size
                       dimensions. Default values are overwritten for specified
                       parameters.
    Available parameters are:
    v_rest: Resting potential. Value range [50, 160].
    v_reset: Reset potential. Value range [50, 160].
    v_thresh: Threshold potential. Value range [50, 220].
    tau_m: Membrane time constant. Value range [0.5, 60]us.
    tau_syn_E: Excitatory synaptic input time constant. Value range
        [0.3, 30]us.
    tau_syn_E: Inhibitory synaptic input time constant. Value range
        [0.3, 30]us.
    cm: Membrane capacitance. Value range [0, 63].
    tau_refrac: Refractory time. Value range [.04, 32]us.
    i_synin_gm_E: Excitatory synaptic input strength bias current. Scales the
        strength of excitatory weights. Technical parameter which needs to be
        same for all populations. Value range [30, 800].
    i_synin_gm_I: Inhibitory synaptic input strength bias current. Scales the
        strength of excitatory weights. Technical parameter which needs to be
        same for all populations. Value range [30, 800].
    synapse_dac_bias: Synapse DAC bias current. Technical parameter which needs
        to be same for all populations Can be lowered in order to reduce the
        amplitude of a spike at the input of the synaptic input OTA. This can
        be useful to avoid saturation when using larger synaptic time
        constants. Value range [30, 1022].

    Attributes:
        calib_target: Archive of cell parameters used for last calibration run.
                      Only available after pynn.preprocess() or pynn.run()
                      call.
        calib_hwparams: Archive of resulting hardware parameters from last
                        calibration run. Only available after pynn.preprocess()
                        or pynn.run() call.
        acutal_hwparams: Hardware parameters used for actual hardware
                         execution, can be manually adjusted. Only available
                         after pynn.preprocess() or pynn.run() call.
    """

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]

    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = False

    default_parameters = {
        'v_rest': 80,
        'v_reset': 70,
        'v_thresh': 125,
        'tau_m': 10,
        'tau_syn_E': 10,
        'tau_syn_I': 10,
        'cm': 63,
        'tau_refrac': 2,
        'i_synin_gm_E': 500,
        'i_synin_gm_I': 500,
        'synapse_dac_bias': 600
    }

    units = {
        'v_rest': 'dimensionless',
        'v_reset': 'dimensionless',
        'v_thresh': 'dimensionless',
        'tau_m': 'us',
        'tau_syn_E': 'us',
        'tau_syn_I': 'us',
        'cm': 'dimensionless',
        'refractory_time': 'us',
        'i_synin_gm_E': 'dimensionless',
        'i_synin_gm_I': 'dimensionless',
        'synapse_dac_bias': 'dimensionless',
        "v": "dimensionless",
        "exc_synin": "dimensionless",
        "inh_synin": "dimensionless",
        "adaptation": "dimensionless"
    }

    translations = build_translations(
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
        ('adaptation', 'adaptation')
    )

    calib_target: ParameterSpace
    calib_hwparams: List[lola.AtomicNeuron]
    actual_hwparams: List[lola.AtomicNeuron]

    # HXNeuron consists of a single compartment with a single circuit
    logical_compartments: Final[halco.LogicalNeuronCompartments] = \
        halco.LogicalNeuronCompartments(
            {halco.CompartmentOnLogicalNeuron():
             [halco.AtomicNeuronOnLogicalNeuron()]})

    @staticmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        return HXNeuron.add_to_network_graph(population, builder)

    @staticmethod
    def add_to_input_generator(population: Population,
                               builder: grenade.InputGenerator):
        pass

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

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
        'adaptation': 'adaptation'
    }

    def add_calib_params(self, calib_params: Dict, cell_ids: List) -> Dict:
        self._calib_target = copy.deepcopy(self.parameter_space)
        paradict = calib_params.__dict__
        for parameters, cell_id in zip(self.parameter_space, cell_ids):
            for param in parameters:
                coord = simulator.state.neuron_placement.id2first_circuit(cell_id)
                myparam = self.param_trans[param]
                if param == "tau_syn_E":
                    paradict[myparam][0][coord] = Quantity(
                        parameters[param], "us")
                elif param == "tau_syn_I":
                    paradict[myparam][1][coord] = Quantity(
                        parameters[param], "us")
                elif param == "synapse_dac_bias":
                    if paradict[param] is not None and\
                            parameters[param] != paradict[param]:
                        raise AttributeError(
                            "synapse_dac_bias requires same value for all "
                            "neurons")
                    paradict[param] = parameters[param]
                elif param == "i_synin_gm_E":
                    if paradict[myparam][0] is not None and\
                            parameters[param] !=\
                            paradict[myparam][0]:
                        raise AttributeError(
                            "i_synin_gm_ex requires same value for all "
                            "neurons")
                    paradict["i_synin_gm"][0] = parameters[param]
                elif param == "i_synin_gm_I":
                    if paradict[myparam][1] is not None and\
                            parameters[param] !=\
                            paradict[myparam][1]:
                        raise AttributeError(
                            "i_synin_gm_in needs to have same value for all "
                            "neurons")
                    paradict[myparam][1] = parameters[param]
                # FIXME handling of Quantity
                elif isinstance(paradict[myparam][coord], Quantity):
                    paradict[myparam][coord] = Quantity(
                        parameters[param], "us")
                else:
                    paradict[myparam][coord] = parameters[param]
        return calib_params

    def add_to_chip(self, cell_ids: List, config: lola.Chip):
        """
        Add configuration of each neuron in the parameter space to the given
        chip object.


        :param cell_ids: Cell IDs for each neuron in the parameter space of
            this celltype object.
        :param chip: Lola chip object which is altered.
        """

        calib_result = []
        for cell_id in cell_ids:

            logical_coord = \
                simulator.state.neuron_placement.id2logicalneuron(cell_id)
            assert len(logical_coord.get_atomic_neurons()) == 1
            coord = logical_coord.get_atomic_neurons()[0]
            atomic_neuron = config.neuron_block.atomic_neurons[coord]

            calib_result.append(
                atomic_neuron)

        self._calib_hwparams = copy.deepcopy(calib_result)
        self._actual_hwparams = calib_result

    @property
    def actual_hwparams(self):
        # cast to tuple prevents overwriting reference with new object
        try:
            return tuple(self._actual_hwparams)
        except AttributeError:
            raise AttributeError("actual_hwparams only available after "
                                 "pynn.preprocess() or pynn.run() call.")

    @property
    def calib_hwparams(self):
        try:
            return self._calib_hwparams
        except AttributeError:
            raise AttributeError("calib_hwparams only available after "
                                 "pynn.preprocess() or pynn.run() call.")

    @property
    def calib_target(self):
        try:
            return self._calib_target
        except AttributeError:
            raise AttributeError("calib_target only available after "
                                 "pynn.preprocess() or pynn.run() call.")


class CalibHXNeuronCoba(CalibHXNeuronCuba):
    """
    HX Neuron with automated calibration. Cell parameters correspond to
    parameters for Calix spiking calibration.

    :param parameters: Mapping of parameters and corresponding values, e.g.
                       dict. Either 1-dimensional or population size
                       dimensions. Default values are overwritten for specified
                       parameters.

    Available parameters are:
    v_rest: Resting potential. Value range [50, 160].
    v_reset: Reset potential. Value range [50, 160].
    v_thresh: Threshold potential. Value range [50, 220].
    tau_m: Membrane time constant. Value range [0.5, 60]us.
    tau_syn_E: Excitatory synaptic input time constant. Value range
        [0.3, 30]us.
    tau_syn_E: Inhibitory synaptic input time constant. Value range
        [0.3, 30]us.
    cm: Membrane capacitance. Value range [0, 63].
    tau_refrac: Refractory time. Value range [.04, 32]us.
    e_rev_E: Excitatory COBA synaptic input reversal potential. At this
        potential, the synaptic input strength will be zero. Value range
        [60, 160].
    e_rev_I: Inhibitory COBA synaptic input reversal potential. At this
        potential, the synaptic input strength will be zero. Value range
        [60, 160].
    i_synin_gm_E: Excitatory synaptic input strength bias current. Scales the
        strength of excitatory weights. Technical parameter which needs to be
        same for all populations. Value range [30, 800].
    i_synin_gm_I: Inhibitory synaptic input strength bias current. Scales the
        strength of excitatory weights. Technical parameter which needs to be
        same for all populations. Value range [30, 800].
    synapse_dac_bias: Synapse DAC bias current. Technical parameter which needs
        to be same for all populations Can be lowered in order to reduce the
        amplitude of a spike at the input of the synaptic input OTA. This can
        be useful to avoid saturation when using larger synaptic time
        constants. Value range [30, 1022].

    Attributes:
        calib_target: Archive of cell parameters used for last calibration run.
                      Only available after pynn.preprocess() or pynn.run()
                      call.
        calib_hwparams: Archive of resulting hardware parameters from last
                        calibration run. Only available after pynn.preprocess()
                        or pynn.run() call.
        acutal_hwparams: Hardware parameters used for actual hardware
                         execution, can be manually adjusted. Only available
                         after pynn.preprocess() or pynn.run() call.
    """

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]
    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = True

    default_parameters = {
        'v_rest': 110,
        'v_reset': 100,
        'v_thresh': 160,
        'tau_m': 10,
        'tau_syn_E': 10,
        'tau_syn_I': 10,
        'cm': 63,
        'tau_refrac': 2,
        'e_rev_E': 300,
        'e_rev_I': 30,
        'i_synin_gm_E': 180,
        'i_synin_gm_I': 250,
        'synapse_dac_bias': 400
    }

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
        "adaptation": "dimensionless"
    }

    translations = build_translations(
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
        ('adaptation', 'adaptation')
    )

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
        'adaptation': 'adaptation'
    }

    def add_calib_params(self, calib_params: Dict, cell_ids: List) -> Dict:
        self._calib_target = copy.deepcopy(self.parameter_space)
        paradict = calib_params.__dict__
        for parameters, cell_id in zip(self.parameter_space, cell_ids):
            for param in parameters:
                coord = simulator.state.neuron_placement.id2first_circuit(cell_id)
                myparam = self.param_trans[param]
                if param == "tau_syn_E":
                    paradict[myparam][0][coord] = Quantity(
                        parameters[param], "us")
                elif param == "tau_syn_I":
                    paradict[myparam][1][coord] = Quantity(
                        parameters[param], "us")
                elif param == "e_rev_E":
                    paradict[myparam][0][coord] = parameters[param]
                elif param == "e_rev_I":
                    paradict[myparam][1][coord] = parameters[param]
                elif param == "synapse_dac_bias":
                    if paradict[param] is not None and\
                            parameters[param] != paradict[param]:
                        raise AttributeError(
                            "synapse_dac_bias requires same value for all "
                            "neurons")
                    paradict[param] = parameters[param]
                elif param == "i_synin_gm_E":
                    if paradict[myparam][0] is not None and\
                            parameters[param] !=\
                            paradict[myparam][0]:
                        raise AttributeError(
                            "i_synin_gm_ex requires same value for all "
                            "neurons")
                    paradict["i_synin_gm"][0] = parameters[param]
                elif param == "i_synin_gm_I":
                    if paradict[myparam][1] is not None and\
                            parameters[param] !=\
                            paradict[myparam][1]:
                        raise AttributeError(
                            "i_synin_gm_in needs to have same value for all "
                            "neurons")
                    paradict[myparam][1] = parameters[param]
                # FIXME handling of Quantity
                elif isinstance(paradict[myparam][coord], Quantity):
                    paradict[myparam][coord] = Quantity(
                        parameters[param], "us")
                else:
                    paradict[myparam][coord] = parameters[param]
        return calib_params


class SpikeSourcePoissonOnChip(StandardCellType, NetworkAddableCell):
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

    _simulator = simulator
    _padi_bus: Optional[halco.PADIBusOnPADIBusBlock] = None

    background_source_clock_frequency: ClassVar[float]

    # TODO: implement L2-based read-out
    recordable = []

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    @staticmethod
    def add_to_network_graph(
            population: Population,
            builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        # register hardware utilisation
        if not population.celltype._padi_bus:
            simulator.state.background_spike_source_placement.register_id(
                list(population.all_cells))
            population.celltype._padi_bus = simulator.state \
                .background_spike_source_placement.id2source(
                    list(population.all_cells))
        # calculate period and rate from rate[Hz]
        assert np.all(population.celltype.parameter_space["rate"]
                      == population.celltype.parameter_space["rate"][0])
        assert np.all(population.celltype.parameter_space["seed"]
                      == population.celltype.parameter_space["seed"][0])
        hwrate = population.celltype.parameter_space["rate"][0] \
            * population.size
        if hwrate > SpikeSourcePoissonOnChip.background_source_clock_frequency:
            raise RuntimeError(
                "The chosen Poisson rate can not be realized on"
                " hardware. The product of rate and number of neurons is too"
                " high in this population.")
        rate = hal.BackgroundSpikeSource.Rate(int(round(
            hal.BackgroundSpikeSource.Rate.max * hwrate
            / SpikeSourcePoissonOnChip.background_source_clock_frequency)))
        prob = (rate.value() + 1) / hal.BackgroundSpikeSource.Rate.size
        period = hal.BackgroundSpikeSource.Period(
            int(round(SpikeSourcePoissonOnChip
                .background_source_clock_frequency / hwrate * prob) - 1))
        # create grenade population
        config = \
            grenade.BackgroundSpikeSourcePopulation.Config()
        config.period = period
        config.rate = rate
        config.seed = hal.BackgroundSpikeSource.Seed(
            population.celltype.parameter_space["seed"][0])
        config.enable_random = True
        # we need both hemispheres because of possibly arbitrary connection
        # targets
        gpopulation = grenade.BackgroundSpikeSourcePopulation(
            population.size,
            {halco.HemisphereOnDLS(0): population.celltype._padi_bus,
             halco.HemisphereOnDLS(1): population.celltype._padi_bus},
            config
        )
        return builder.add(gpopulation)

    @staticmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):
        pass


SpikeSourcePoissonOnChip.background_source_clock_frequency = \
    sta.DigitalInit() \
    .adplls[sta.DigitalInit().pll_clock_output_block.get_clock_output(
        halco.PLLClockOutputOnDLS.phy_ref_clk).select_adpll] \
    .calculate_output_frequency(
        sta.DigitalInit().pll_clock_output_block.get_clock_output(
            halco.PLLClockOutputOnDLS.phy_ref_clk
        ).select_adpll_output) / 2.  # spl1_clk


class SpikeSourcePoisson(cells.SpikeSourcePoisson, NetworkAddableCell):
    """
    Spike source, generating spikes according to a Poisson process.
    """

    def __init__(self, start, rate, duration):

        parameters = {"start": start,
                      "rate": rate,
                      "duration": duration}
        super(cells.SpikeSourcePoisson, self).__init__(**parameters)

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

    # TODO: implement L2-based read-out injected spikes
    recordable = []

    def can_record(self, variable: str) -> bool:
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

            self._used_parameters = dict(start=np.zeros(pop_size),
                                         rate=np.zeros(pop_size),
                                         duration=np.zeros(pop_size))
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
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        # create grenade population
        gpopulation = grenade.ExternalPopulation(
            population.size)
        # add to builder
        return builder.add(gpopulation)

    @staticmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):

        spiketimes = population.celltype.get_spike_times()
        spiketimes = [np.sort(spiketimes_neuron) for spiketimes_neuron
                      in spiketimes]
        descriptor = grenade.PopulationDescriptor(
            simulator.state.populations.index(population))
        builder.add(spiketimes, descriptor)


class SpikeSourceArray(cells.SpikeSourceArray, NetworkAddableCell):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    # FIXME: workaround, see https://github.com/NeuralEnsemble/PyNN/issues/709
    default_parameters = {'spike_times': ArrayParameter([])}

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )

    # TODO: implement L2-based read-out for injected spikes
    recordable = []

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    @staticmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        # create grenade population
        gpopulation = grenade.ExternalPopulation(
            population.size)
        # add to builder
        return builder.add(gpopulation)

    @staticmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):
        spiketimes = population.celltype.parameter_space["spike_times"]
        spiketimes = [s.value for s in spiketimes]
        descriptor = grenade.PopulationDescriptor(
            simulator.state.populations.index(population))
        builder.add(spiketimes, descriptor)
