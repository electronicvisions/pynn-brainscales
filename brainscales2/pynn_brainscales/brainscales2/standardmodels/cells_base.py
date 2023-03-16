from abc import abstractmethod, ABC

from pyNN.common import Population
import pygrenade_vx.network.placed_logical as grenade


class NetworkAddableCell(ABC):
    @staticmethod
    @abstractmethod
    def add_to_network_graph(population: Population,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
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
