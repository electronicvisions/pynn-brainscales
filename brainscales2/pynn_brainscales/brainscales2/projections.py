from __future__ import annotations
from typing import List
from copy import deepcopy
import pyNN.common
from pyNN.common import Population, PopulationView, Assembly
from pyNN.space import Space
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2 import simulator
import pygrenade_vx as grenade


class Projection(pyNN.common.Projection):
    _simulator = simulator
    _static_synapse_class = StaticSynapse

    # pylint: disable=too-many-arguments
    def __init__(self, presynaptic_neurons, postsynaptic_neurons, connector,
                 synapse_type=None, source=None, receptor_type=None,
                 space=Space(), label=None):
        """
        Create a new projection, connecting the pre- and post-synaptic neurons.

        :param presynaptic_neurons: Population, PopulationView or
                                    Assembly object.

        :param postsynaptic_neurons: Population, PopulationView or
                                     Assembly object.

        :param connector: a Connector object, encapsulating the algorithm to
                          use for connecting the neurons.

        :param synapse_type: a SynapseType object specifying which synaptic
                             connection mechanisms to use,
                             defaults to None

        :param source: string specifying which attribute of the presynaptic
                       cell signals action potentials. This is only needed for
                       multicompartmental cells with branching axons or
                       dendrodendritic synapses. All standard cells have a
                       single source, and this is the default,
                       defaults to None

        :param receptor_type: string specifying which synaptic receptor_type
                              type on the postsynaptic cell to connect to. For
                              standard cells, this can be 'excitatory' or
                              'inhibitory'. For non-standard cells, it could be
                              'NMDA', etc. If receptor_type is not  given, the
                              default values of 'excitatory' is used,
                              defaults to None

        :param space: Space object, determining how distances should be
                      calculated for distance-dependent wiring schemes or
                      parameter values,
                      defaults to Space()

        :param label: a name for the projection (one will be auto-generated
                      if this is not supplied),
                      defaults to None
        """
        super(Projection, self).__init__(presynaptic_neurons,
                                         postsynaptic_neurons, connector,
                                         synapse_type, source, receptor_type,
                                         space, label)
        self.connections = []
        connector.connect(self)
        self._simulator.state.projections.append(self)
        self.changed_since_last_run = True

    def __len__(self):
        """Return the total number of local connections."""
        return len(self.connections)

    def __getitem__(self, i):
        """Return the *i*th connection within the Projection."""
        return self.connections[i]

    # pylint: disable=protected-access
    def _set_attributes(self, parameter_space):
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))
        for name, value in parameter_space.items():
            for pre in self.pre:
                pre = self.pre.id_to_index(pre)
                for post in self.post:
                    post = self.post.id_to_index(post)
                    setattr(self.connections[pre + len(self.pre) * post],
                            name, float(value[pre][post]))

    def _set_initial_value_array(self, variable, initial_value):
        raise NotImplementedError

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.
        """
        assert len(presynaptic_indices) > 0 and \
            len(presynaptic_indices) <= len(self.pre.all_cells)
        for pre_ind in presynaptic_indices:
            assert pre_ind < self._simulator.state.id_counter
        assert postsynaptic_index < self._simulator.state.id_counter

        for pre_index in presynaptic_indices:
            filtered_connection_parameters = deepcopy(connection_parameters)
            for key, value in connection_parameters.items():
                if not isinstance(value, float) or isinstance(value, int) \
                        and len(value) > 0:
                    filtered_connection_parameters[key] = \
                        value[pre_index]

            if filtered_connection_parameters["weight"] < 0 \
                    or filtered_connection_parameters["weight"] \
                    > self._simulator.state.max_weight:
                raise ValueError(
                    "The weight must be positive and smaller than {}."
                    .format(self._simulator.state.max_weight))

            connection = Connection(self, pre_index, postsynaptic_index,
                                    **filtered_connection_parameters)
            self.connections.append(connection)

    @staticmethod
    def add_to_network_graph(populations: List[Population],
                             projection: Projection,
                             builder: grenade.NetworkBuilder) \
            -> grenade.ProjectionDescriptor:

        if isinstance(projection.pre, Assembly):
            raise NotImplementedError("Assemblies are not supported yet")
        if isinstance(projection.post, Assembly):
            raise NotImplementedError("Assemblies are not supported yet")

        # grenade has no concept of pop views, we therefore need to
        # get pre- and post-synaptic population descriptor of the parent in
        # case of pop views
        pre_is_view = isinstance(projection.pre, PopulationView)
        post_is_view = isinstance(projection.post, PopulationView)
        pre = projection.pre.grandparent if \
            pre_is_view else projection.pre
        post = projection.post.grandparent if \
            post_is_view else projection.post

        population_pre = grenade.PopulationDescriptor(
            populations.index(pre))
        population_post = grenade.PopulationDescriptor(
            populations.index(post))

        connections: grenade.Projection.Connections = [
            grenade.Projection.Connection(
                conn.pop_pre_index,
                conn.pop_post_index,
                int(conn.weight))
            for conn in projection.connections]

        if projection.receptor_type == "excitatory":
            receptor_type = grenade.Projection.ReceptorType.excitatory
        elif projection.receptor_type == "inhibitory":
            receptor_type = grenade.Projection.ReceptorType.inhibitory
        else:
            raise NotImplementedError(
                "grenade.Projection.RecetorType does "
                + f"not support {projection.receptor_type}.")

        gprojection = grenade.Projection(
            receptor_type, connections, population_pre, population_post)

        return builder.add(gprojection)


class Connection(pyNN.common.Connection):
    """
    Store an individual plastic connection and information about it.
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, projection, pre_index, post_index, **parameters):
        self.projection = projection
        self.presynaptic_index = pre_index
        self.changed_since_last_run = True

        if isinstance(projection.pre, PopulationView):
            self.pop_pre_index = \
                projection.pre.index_in_grandparent([pre_index])[0]
        else:
            self.pop_pre_index = pre_index

        self.postsynaptic_index = post_index
        if isinstance(projection.post, PopulationView):
            self.pop_post_index = \
                projection.post.index_in_grandparent([post_index])[0]
        else:
            self.pop_post_index = post_index
        self._weight = parameters["weight"]
        if parameters["delay"] != 0:
            raise ValueError("Setting the delay unequal 0 is not supported.")
        self._delay = parameters["delay"]
        self.parameters = {x: parameters[x] for x in parameters
                           if x not in ["delay", "weight"]}

    def _set_weight(self, new_weight):
        new_weight = round(new_weight)
        if new_weight < 0 or new_weight > simulator.state.max_weight:
            raise ValueError("The weight must be in the interval [0, {}]."
                             .format(simulator.state.max_weight))
        self._weight = new_weight
        self.projection.changed_since_last_run = True

    def _get_weight(self):
        return self._weight

    def _set_delay(self, new_delay):
        if new_delay != 0:
            raise ValueError("Setting the delay unequal 0 is not supported.")
        self._delay = new_delay
        self.projection.changed_since_last_run = True

    def _get_delay(self):
        return self._delay

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

    def as_tuple(self, *attribute_names):
        return tuple(getattr(self, name) for name in attribute_names)
