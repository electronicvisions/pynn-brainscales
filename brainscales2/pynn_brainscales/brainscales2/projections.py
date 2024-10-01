from __future__ import annotations
from copy import deepcopy
from typing import List
import math
import numpy as np
import pyNN.common
import pyNN.errors
from pyNN.common import PopulationView, Assembly
from pyNN.space import Space
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.plasticity_rules import PlasticityRuleHandle
import pygrenade_vx.network as grenade
import pygrenade_common as grenade_common
from dlens_vx_v3 import halco


class Projection(
        pyNN.common.Projection,
        grenade.abstract.frontend.ExperimentElement):
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

        grenade.abstract.frontend.ExperimentElement.__init__(
            self, self._simulator.state.grenade_experiment)
        self.grenade_descriptor = None
        self.connections = []

        try:
            connector.connect(self)
        except pyNN.errors.ConnectionError as err:
            self._simulator.state.grenade_experiment.elements.remove(self)
            raise err

        def key(connection):
            return (connection.presynaptic_index,
                    connection.postsynaptic_index)
        # PyNN creates connections in column-major order, we require
        # row-major order.
        self.connections = sorted(self.connections, key=key)

        self._simulator.state.projections.append(self)

        # determine from which to which compartment the connection will be
        # established
        self.pre_compartment = halco.CompartmentOnLogicalNeuron()
        if connector.source_location_selector is not None:
            pre = getattr(self.pre, "grandparent", self.pre)
            self.pre_compartment = \
                self._get_comp_id_from_location(
                    connector.source_location_selector, pre.celltype)
        self.post_compartment = halco.CompartmentOnLogicalNeuron()
        if connector.location_selector is not None:
            post = getattr(self.post, "grandparent", self.post)
            self.post_compartment = \
                self._get_comp_id_from_location(
                    connector.location_selector,
                    post.celltype)

    @staticmethod
    def _get_comp_id_from_location(location: str, celltype
                                   ) -> halco.CompartmentOnLogicalNeuron:
        try:
            comp_ids = celltype.get_compartment_ids([location])
        except AttributeError as err:
            raise ValueError(
                'Can not extract recording locations for celltype '
                f'"{celltype.__class__.__name__}".') from err
        if len(comp_ids) == 0:
            raise ValueError(f'Label "{location}" does not exist.')
        if len(comp_ids) > 1:
            raise ValueError(f'Label "{location}" matches more than one '
                             'compartment.')
        return comp_ids[0]

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
                            location_selector=None, **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.
        """
        del location_selector  # we handle "location_selector" in __init__
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

    def _generate_parameterization(self) \
            -> grenade_common.Computation.Parameterization:
        return grenade.abstract.UncalibratedSynapse.ParameterSpace\
            .Parameterization([
                grenade.abstract.UncalibratedSynapse.Weight(
                    int(abs(c.weight)))
                for c in self.connections])

    # pylint: disable=too-many-locals
    def add_to_topology(
            self,
            experiment: grenade.abstract.frontend.ExperimentSnippet):
        if isinstance(self.pre, Assembly):
            raise NotImplementedError("Assemblies are not supported yet")

        # grenade has no concept of pop views, we therefore need to
        # get pre- and post-synaptic population descriptor of the parent in
        # case of pop views
        pre = getattr(self.pre, "grandparent", self.pre)

        # grenade has no concept of pop views, we therefore need to
        # get pre- and post-synaptic population descriptor of the parent in
        # case of pop views
        post = getattr(self.post, "grandparent", self.post)

        if pre.grenade_descriptor is None:
            return False
        if post.grenade_descriptor is None:
            return False

        # generate vertex
        pre_dimension = 0
        post_dimension = 1
        connections_points = np.empty((len(self.connections), 2), dtype=int)
        for i, conn in enumerate(self.connections):
            connections_points[i, pre_dimension] = conn.pop_pre_index
            connections_points[i, post_dimension] = conn.pop_post_index
        connections_sequence = grenade_common.ListMultiIndexSequence([])
        connections_sequence.from_numpy(
            connections_points,
            [grenade_common.CellOnPopulationDimensionUnit(),
             grenade_common.CellOnPopulationDimensionUnit()])

        input_sequence = connections_sequence\
            .distinct_projection({pre_dimension})
        output_sequence = connections_sequence\
            .distinct_projection({post_dimension})

        vertex = grenade_common.Projection(
            synapse=grenade.abstract.UncalibratedSynapse(),
            parameter_space=grenade.abstract.UncalibratedSynapse
            .ParameterSpace([
                grenade.abstract.UncalibratedSynapse.Weight(
                    Connection.round_up_63(int(abs(c.weight))))
                for c in self.connections]),
            connector=grenade_common.SequenceConnector(
                input_sequence,
                output_sequence,
                connections_sequence),
            time_domain=grenade_common.TimeDomainOnTopology())

        if self.grenade_descriptor is not None and \
                experiment.topology.contains(self.grenade_descriptor):
            experiment.topology.clear_vertex(self.grenade_descriptor)
            experiment.topology.set(self.grenade_descriptor, vertex)
        else:
            self.grenade_descriptor = experiment.topology.add_vertex(vertex)

        # add in-edge
        in_edge = grenade_common.Edge(
            input_sequence.cartesian_product(
                grenade_common.ListMultiIndexSequence([
                    grenade_common.MultiIndex([int(self.pre_compartment)])],
                    [grenade_common.CompartmentOnNeuronDimensionUnit()])),
            input_sequence
        )
        experiment.topology.add_edge(
            pre.grenade_descriptor,
            self.grenade_descriptor,
            in_edge)

        # add out-edge
        receptor_on_compartment = post.celltype.get_receptor(
            self.receptor_type, self.post_compartment)

        post_compartment_on_neuron = grenade_common.CuboidMultiIndexSequence(
            [1],
            grenade_common.MultiIndex([int(self.post_compartment)]),
            [grenade_common.CompartmentOnNeuronDimensionUnit()])

        out_edge = grenade_common.Edge(
            output_sequence,
            output_sequence
            .cartesian_product(post_compartment_on_neuron)
            .cartesian_product(receptor_on_compartment)
        )
        experiment.topology.add_edge(
            self.grenade_descriptor,
            post.grenade_descriptor,
            out_edge)

        return True

    def add_to_input_data(
            self,
            experiment: grenade.abstract.frontend.ExperimentSnippet,
            snippet_begin_time: float,
            snippet_end_time: float):
        experiment.input_data.ports.set(
            (self.grenade_descriptor, 1),
            self._generate_parameterization())

    def extract_output_data(self, experiment):
        pass

    @property
    def placed_connections(self) -> List[List[halco.SynapseOnDLS]]:
        """
        Query the last routing run for placement of this projection.
        """
        if self.grenade_descriptor is None:
            raise RuntimeError(
                "placed_connections requires a previous routing run"
                ", which is executed on pynn.run().")
        return grenade.abstract.reverse_mapping\
            .get_uncalibrated_synapse_coordinates(
                self.grenade_descriptor,
                self._simulator.state.grenade_experiment.snippets[-1]
                .mapped_topology)

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

        if self.synapse_type.plasticity_rule._recording_data is None:
            raise RuntimeError(
                "Plasticity rule observables only available after execution.")
        observable_data = []
        for snippet in self.synapse_type.plasticity_rule._recording_data:  # pylint: disable=protected-access
            projection_on_plasticity_rule = self.synapse_type.plasticity_rule\
                ._projections.index(self)  # pylint: disable=protected-access
            if snippet is not None and observable in snippet.data_per_synapse:
                observable_data.append(
                    snippet.data_per_synapse[observable][
                        projection_on_plasticity_rule][
                        simulator.state.batch_entry])
            else:
                observable_data.append(None)

        return observable_data


class Connection(pyNN.common.Connection):
    """
    Store an individual plastic connection and information about it.
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, projection, pre_index, post_index, **parameters):
        self.projection = projection
        self.presynaptic_index = pre_index
        self._weight = None

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

    @staticmethod
    def round_up_63(value):
        """
        Grenade expects the maximum weight the user wants to set.
        We currently do not implement a maximum weight in PyNN.
        Therefore, we round it up to the maximum weight possible with
        the given number of synapses needed to realize the given value.
        """
        return max(1, math.ceil(abs(value) / 63)) * 63

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
        old_weight_max = None
        if self._weight is not None:
            old_weight_max = Connection.round_up_63(self._weight)
        self._weight = new_weight
        new_weight_max = Connection.round_up_63(new_weight)
        self.projection.changed_topology = self.projection.changed_topology \
            or (new_weight_max != old_weight_max)
        self.projection.changed_input_data = True

    def _get_weight(self):
        return self._weight

    def _set_delay(self, new_delay):
        if new_delay != 0:
            raise ValueError("Setting the delay unequal 0 is not supported.")
        self._delay = new_delay
        self.projection.changed_input_data = True
        self.projection.changed_topology = True

    def _get_delay(self):
        return self._delay

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

    def as_tuple(self, *attribute_names):
        return tuple(getattr(self, name) for name in attribute_names)
