from __future__ import annotations
from typing import List
from copy import deepcopy
import numpy as np
import pyNN.common
from pyNN.common import Population, PopulationView, Assembly
from pyNN.space import Space
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.plasticity_rules import PlasticityRuleHandle
import pygrenade_vx.network as grenade


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
        super().__init__(presynaptic_neurons, postsynaptic_neurons, connector,
                         synapse_type, source, receptor_type, space, label)
        self.connections = []
        connector.connect(self)

        def key(connection):
            return (connection.presynaptic_index,
                    connection.postsynaptic_index)
        # PyNN creates connections in column-major order, we require
        # row-major order.
        self.connections = sorted(self.connections, key=key)

        self._simulator.state.projections.append(self)
        self.changed_since_last_run = True

    def __setattr__(self, name, value):
        # Handle (de-)registering of projection in plasticity rule.
        # A plasticity rule can be applied to multiple projections
        # and then serve handles to all in the kernel code, for which
        # registration here is required.
        if name == "synapse_type":
            if hasattr(self, name):
                if isinstance(self.synapse_type, PlasticityRuleHandle) \
                        and self.synapse_type.plasticity_rule is not None:
                    self.synapse_type.plasticity_rule._remove_projection(self)
            super().__setattr__(name, value)
            if isinstance(self.synapse_type, PlasticityRuleHandle) \
                    and self.synapse_type.plasticity_rule is not None:
                self.synapse_type.plasticity_rule._add_projection(self)
        else:
            super().__setattr__(name, value)

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
            for connection in self.connections:
                setattr(connection, name, float(value[
                    connection.presynaptic_index][
                        connection.postsynaptic_index]))

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
        pre_has_grandparent = hasattr(projection.pre, "grandparent")
        post_has_grandparent = hasattr(projection.post, "grandparent")
        pre = projection.pre.grandparent if \
            pre_has_grandparent else projection.pre
        post = projection.post.grandparent if \
            post_has_grandparent else projection.post

        population_pre = grenade.PopulationDescriptor(
            populations.index(pre))
        population_post = grenade.PopulationDescriptor(
            populations.index(post))

        connections = np.empty((len(projection.connections), 5), dtype=int)
        for i, conn in enumerate(projection.connections):
            connections[i, 0] = conn.pop_pre_index
            connections[i, 1] = 0
            connections[i, 2] = conn.pop_post_index
            connections[i, 3] = 0
            connections[i, 4] = int(abs(conn.weight))

        if projection.receptor_type == "excitatory":
            receptor_type = \
                grenade.Receptor(
                    grenade.Receptor.ID(),
                    grenade.Receptor.Type.excitatory)
        elif projection.receptor_type == "inhibitory":
            receptor_type = \
                grenade.Receptor(
                    grenade.Receptor.ID(),
                    grenade.Receptor.Type.inhibitory)
        else:
            raise NotImplementedError(
                "grenade.Projection.RecetorType does "
                + f"not support {projection.receptor_type}.")

        gprojection = grenade.Projection()
        gprojection.from_numpy(
            receptor_type, connections, population_pre, population_post)

        return builder.add(gprojection)

    @property
    def placed_connections(self):
        """
        Query the last routing run for placement of this projection.
        """
        if self._simulator.state.grenade_network_graph is None:
            raise RuntimeError(
                "placed_connections requires a previous routing run"
                ", which is executed on pynn.run().")
        return self._simulator.state.grenade_network_graph \
            .get_placed_connections(grenade.ProjectionDescriptor(
                self._simulator.state.projections.index(self)))

    def get_data(self, observable: str):
        """
        Get data for an observable per synapse.

        :param observable: Name of observable.
        :return: Array with recorded data. The array's entries are values
            for each timer entry. Each value has a `.data` attribute,
            containing the recorded data.

        :raises RuntimeError: If observable name is not known or
            the projection does not implement a plasticity rule.
        """

        if not isinstance(self.synapse_type, PlasticityRuleHandle):
            raise RuntimeError("Synapse type can't have observables, since it"
                               + " is not derived from PlasticityRuleHandle.")
        if self.synapse_type.plasticity_rule is None:
            raise RuntimeError("Synapse type can't have observables, since it"
                               + " does not hold a plasticity rule.")
        if observable not in self.synapse_type.plasticity_rule.observables:
            raise RuntimeError(
                "Synapse type doesn't have requested observable.")
        return self._simulator.state.synaptic_observables[
            self._simulator.state.projections.index(self)][observable]


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
        if ((parameters["weight"] < 0)
                and ((self.projection.post.conductance_based
                      or self.projection.receptor_type != "inhibitory"))) or \
                ((parameters["weight"] > 0)
                    and (self.projection.receptor_type != "excitatory")):
            raise pyNN.errors.ConnectionError(
                "Weights must be positive for conductance-based and/or "
                "excitatory synapses and negative for inhibitory synapses")
        self._set_weight(parameters["weight"])
        if parameters["delay"] != 0:
            raise ValueError("Setting the delay unequal 0 is not supported.")
        self._delay = parameters["delay"]
        self.parameters = {x: param for x, param in parameters.items()
                           if x not in ["delay", "weight"]}

    def _set_weight(self, new_weight):
        new_weight = round(new_weight)
        if ((new_weight < 0)
                and ((self.projection.post.conductance_based
                      or self.projection.receptor_type != "inhibitory"))) or \
                ((new_weight > 0)
                    and (self.projection.receptor_type != "excitatory")):
            raise pyNN.errors.ConnectionError(
                "Weights must be positive for conductance-based and/or "
                "excitatory synapses and negative for inhibitory synapses")
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
