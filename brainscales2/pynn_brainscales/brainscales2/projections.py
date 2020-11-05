from copy import deepcopy
import pyNN.common
from pyNN.space import Space
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2 import simulator


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

        presynaptic_cells = self.pre[presynaptic_indices]
        postsynaptic_cell = self.post[postsynaptic_index]

        for pre_cell in presynaptic_cells:
            filtered_connection_parameters = deepcopy(connection_parameters)
            for key, value in connection_parameters.items():
                if not isinstance(value, float) or isinstance(value, int) \
                        and len(value) > 0:
                    filtered_connection_parameters[key] = \
                        value[self.pre.id_to_index(pre_cell)]

            if filtered_connection_parameters["weight"] < 0 \
                    or filtered_connection_parameters["weight"] \
                    > self._simulator.state.max_weight:
                raise ValueError(
                    "The weight must be positive and smaller than {}."
                    .format(self._simulator.state.max_weight))

            connection = Connection(self, pre_cell, postsynaptic_cell,
                                    **filtered_connection_parameters)
            self.connections.append(connection)
            self._simulator.state.connections.append(connection)


class Connection(pyNN.common.Connection):
    """
    Store an individual plastic connection and information about it.
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, projection, pre_cell, post_cell, **parameters):
        self.projection = projection
        # TODO: understand why lookup via int is faster, cf. #3749
        self.presynaptic_index = projection.pre.id_to_index(int(pre_cell))
        self.postsynaptic_index = projection.post.id_to_index(int(post_cell))
        self.presynaptic_cell = \
            pre_cell.parent.all_cells[self.presynaptic_index]
        self.postsynaptic_cell = post_cell
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

    def _get_weight(self):
        return self._weight

    def _set_delay(self, new_delay):
        if new_delay != 0:
            raise ValueError("Setting the delay unequal 0 is not supported.")
        self._delay = new_delay

    def _get_delay(self):
        return self._delay

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

    def as_tuple(self, *attribute_names):
        return tuple(getattr(self, name) for name in attribute_names)
