from abc import abstractmethod, ABC
from typing import Optional, Dict, Any, Final
from pyNN.common import Population
from pyNN.standardmodels import build_translations
from pyNN.standardmodels import StandardCellType \
    as UpstreamStandardCellType
import pygrenade_vx.network as grenade
from pynn_brainscales.brainscales2 import plasticity_rules
from dlens_vx_v3 import lola, halco


class StandardCellType(ABC, UpstreamStandardCellType):
    """
    Network addable standard cell type, to be used as base for all cells.
    """

    def __init__(self, **parameters):
        UpstreamStandardCellType.__init__(self, **parameters)

    @staticmethod
    @abstractmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationOnNetwork:
        """
        Add population to network builder.
        :param population: Population to add featuring this cell's celltype.
        :param builder: Network builder to add population to.
        :return: Descriptor of added population
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def add_to_input_generator(
            population: Population,
            builder: grenade.InputGenerator):
        """
        Add external events to input generator.
        :param population: Population to add featuring this cell's celltype.
        :param builder: Input builder to add external events to.
        """
        raise NotImplementedError


class NeuronCellType(StandardCellType, plasticity_rules.PlasticityRuleHandle):
    """
    Network addable cell with plasticity rule handle.

    Currently this includes setting the readout source of each neuron
    to be available to the plasticity rule.
    """

    ReadoutSource = lola.AtomicNeuron.Readout.Source

    translations: Final[Dict[str, str]] = build_translations(
        ("plasticity_rule_readout_source",
         "plasticity_rule_readout_source"),
        ("plasticity_rule_enable_readout_source",
         "plasticity_rule_enable_readout_source"),
    )

    default_parameters: Final[Dict[str, Any]] = {
        "plasticity_rule_readout_source":
        float(lola.AtomicNeuron.Readout.Source.membrane),
        "plasticity_rule_enable_readout_source": False}

    units: Final[Dict[str, str]] = {
        "plasticity_rule_readout_source": "dimensionless",
        "plasticity_rule_enable_readout_source": "dimensionless"
    }

    def __init__(
            self,
            plasticity_rule: Optional[plasticity_rules.PlasticityRule] = None,
            **parameters):
        plasticity_rules.PlasticityRuleHandle.__init__(
            self, plasticity_rule=plasticity_rule)
        StandardCellType.__init__(self, **parameters)

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
