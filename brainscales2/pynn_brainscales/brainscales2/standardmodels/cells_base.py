from abc import abstractmethod, ABC
from typing import Optional, Dict, Any, Final
from pyNN.common import Population
from pyNN.standardmodels import build_translations
from pyNN.standardmodels import StandardCellType \
    as UpstreamStandardCellType
import pygrenade_common as grenade
import pygrenade_vx as grenade_vx
import pygrenade_vx.network.abstract as grenade_abstract
from pynn_brainscales.brainscales2 import plasticity_rules
from pynn_brainscales.brainscales2.recording_data import ObservableType
from dlens_vx_v3 import lola


class StandardCellType(ABC, UpstreamStandardCellType):
    """
    Network addable standard cell type, to be used as base for all cells.
    """

    # DimensionUnit of how recording sites are defined in grenade for this
    # cell type
    recording_site_dimension_unit = \
        grenade_abstract.AtomicNeuronOnCompartmentDimensionUnit()

    spike_port: int = 0
    analog_obs_port: int = 1

    def __init__(self, **parameters):
        # only forward non None values (parameter spaces can not handle None
        # values)
        UpstreamStandardCellType.__init__(
            self,
            **{name: value for name, value in parameters.items()
               if value is not None})

    @staticmethod
    @abstractmethod
    def generate_vertex(population: Population) \
            -> grenade.Population:
        """
        Generate vertex representation for this population.
        :param population: Population featuring this cell's celltype
        :return: Population vertex
        """
        raise NotImplementedError

    @abstractmethod
    def generate_input_data(
            self,
            population: Population,
            experiment: grenade_vx.network.abstract.frontend
            .ExperimentSnippet,
            snippet_begin_time, snippet_end_time) \
            -> Dict[int, grenade.PortData]:
        """
        Generate input data for this population.
        :param population: Population featuring this cell's celltype
        :param experiment: Experiment snippet to generate data for
        :param snippet_begin_time: Begin time of snippet
        :param snippet_end_time: End time of snippet
        :return: Population input data
        """
        raise NotImplementedError

    def get_receptor(
            self, name: str, compartment: grenade.CompartmentOnNeuron)\
            -> grenade.MultiIndexSequence:  # pylint: disable=unused-argument
        """
        Get receptor by name for compartment.
        :param name: Name of receptor
        :param compartment: Compartment identifier
        """
        receptor = None
        if name == "excitatory":
            receptor = grenade.ReceptorOnCompartment(0)
        elif name == "inhibitory":
            receptor = grenade.ReceptorOnCompartment(1)
        else:
            raise NotImplementedError("Receptor not implemented.")
        return grenade.CuboidMultiIndexSequence(
            [1], grenade.MultiIndex([receptor.value()]),
            [grenade.ReceptorOnCompartmentDimensionUnit()])

    def validate_parameter_space(self):
        """
        Raise if the parameter space is not yet valid.

        In case an initial config is supplied upon setup,
        the parameter space is not valid until mapping
        for uncalibrated neurons.
        """

    def get_recording_site(
        self,
        name: str,
        compartment: grenade.CompartmentOnNeuron
    ) -> int:
        """
        Get recording site by name.

        :param name: Name of observable/mechanism which should be recorded.
        :param compartment: Compartment identifier
        :return: ID of the neuron circuit/mechanism which corresponds to
            the recorded observable.
        """
        del compartment

        if name not in self.recordable:
            raise NotImplementedError(
                "The given observable can not be recorded. This celltype "
                f"supports: {self.recordable}")
        # we always record observables in the first neuron circuit
        return 0

    @classmethod
    def get_observable_type(cls,
                            name: str,
                            compartment: grenade.CompartmentOnNeuron
                            ) -> ObservableType:
        """
        Get the type of the observable.

        :param name: Name of variable for which to get the observable type.
        :param compartment: Compartment identifier
        :return: Type of observable.
        """
        del compartment

        if name not in cls.recordable:
            raise RuntimeError(f"Unknown variable '{name}'.")

        if name in ['v', 'exc_synin', 'inh_synin', 'adaptation']:
            return ObservableType.ANALOG
        if name in ['spikes']:
            return ObservableType.EVENT

        raise RuntimeError(f"Observable type not spcified '{name}'.")


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

    # pylint: disable-next=invalid-name
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


class ExternalNeuron(StandardCellType):
    """
    class used to identify external neurons
    external neurons should inherit from this class
    """
