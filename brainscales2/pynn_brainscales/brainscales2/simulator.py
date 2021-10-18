import time
import itertools
from copy import copy
from typing import Optional, Final, List, Dict, Union, Set
import numpy as np
from pyNN.common import IDMixin, Population, Projection
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells import HXNeuron, \
    SpikeSourceArray, SpikeSourcePoisson, SpikeSourcePoissonOnChip
from dlens_vx_v3 import hal, halco, sta, lola, logger
import pygrenade_vx as grenade


name = "HX"  # for use in annotating output data


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        int.__init__(n)
        IDMixin.__init__(self)


class NeuronPlacement:
    # TODO: support multi compartment issue #3750
    """
    Tracks assignment of pyNN IDs of HXNeuron based populations to the
    corresponding hardware entity, i.e. AtomicNeuronOnDLS. Default constructed
    with 1 to 1 permutation.

    :param neuron_id: Look up table for permutation. Index: HW related
                    population neuron enumeration. Value: HW neuron
                    enumeration.
    """
    _id_2_an: Dict[ID, halco.AtomicNeuronOnDLS]
    _permutation: List[halco.AtomicNeuronOnDLS]
    _max_num_entries: Final[int] = halco.AtomicNeuronOnDLS.size
    default_permutation: Final[List[int]] = range(halco.AtomicNeuronOnDLS.size)

    def __init__(self, permutation: List[int] = None):
        if permutation is None:
            permutation = range(self._max_num_entries)
        self._id_2_an = dict()
        self._permutation = self._check_and_transform(permutation)

    def register_id(self, neuron_id: Union[List[ID], ID]):
        """
        Register a new ID to placement

        :param neuron_id: pyNN neuron ID to be registered
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if len(self._id_2_an) + len(neuron_id) > len(self._permutation):
            raise ValueError(
                f"Cannot register more than {len(self._permutation)} IDs")
        for idx in neuron_id:
            self._id_2_an[idx] = self._permutation[len(self._id_2_an)]

    def id2atomicneuron(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[halco.AtomicNeuronOnDLS], halco.AtomicNeuronOnDLS]:
        """
        Get hardware coordinate from pyNN ID

        :param neuron_id: pyNN neuron ID
        """
        try:
            return [self._id_2_an[idx] for idx in neuron_id]
        except TypeError:
            return self._id_2_an[neuron_id]

    def id2hwenum(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[int], int]:
        """
        Get hardware coordinate as plain int from pyNN ID

        :param neuron_id: pyNN neuron ID
        """
        atomic_neuron = self.id2atomicneuron(neuron_id)
        try:
            return [int(idx.toEnum()) for idx in atomic_neuron]
        except TypeError:
            return int(atomic_neuron.toEnum())

    @staticmethod
    def _check_and_transform(lut: list) -> list:

        cell_id_size = NeuronPlacement._max_num_entries
        if len(lut) > cell_id_size:
            raise ValueError("Too many elements in HW LUT.")
        if len(lut) > len(set(lut)):
            raise ValueError("Non unique entries in HW LUT.")
        permutation = []
        for neuron_idx in lut:
            if not 0 <= neuron_idx < cell_id_size:
                raise ValueError(
                    f"NeuronPermutation list entry {neuron_idx} out of range. "
                    + f"Needs to be in range [0, {cell_id_size - 1}]"
                )
            coord = halco.AtomicNeuronOnDLS(halco.common.Enum(neuron_idx))
            permutation.append(coord)
        return permutation


class BackgroundSpikeSourcePlacement:
    """
    Tracks assignment of pyNN IDs of SpikeSourcePoissonOnChip based populations
    to the corresponding hardware entity, i.e. BackgroundSpikeSourceOnDLS. We
    use one source on each hemisphere to ensure arbitrary routing works.
    Default constructed with reversed 1 to 1 permutation to yield better
    distribution for small networks.

    :cvar default_permutation: Default permutation, where allocation is ordered
                               to start at the highest-enum PADI-bus to reduce
                               overlap with allocated neurons.
    """
    _pb_2_id: Dict[halco.PADIBusOnPADIBusBlock, List[ID]]
    _permutation: List[halco.PADIBusOnPADIBusBlock]
    _max_num_entries: Final[int] = halco.PADIBusOnPADIBusBlock.size
    default_permutation: Final[List[int]] = list(reversed(range(
        halco.PADIBusOnPADIBusBlock.size)))

    def __init__(self, permutation: List[int] = None):
        """
        :param permutation: Look up table for permutation. Index: HW related
                            population neuron enumeration. Value: HW neuron
                            enumeration.
        """

        if permutation is None:
            permutation = self.default_permutation
        self._pb_2_id = dict()
        self._permutation = self._check_and_transform(permutation)

    def register_id(self, neuron_id: Union[List[ID], ID]):
        """
        Register a new ID to placement

        :param neuron_id: pyNN neuron ID to be registered
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if len(self._pb_2_id) + 1 > len(self._permutation):
            raise ValueError(
                f"Cannot register more than {len(self._permutation)} ID sets")
        self._pb_2_id[self._permutation[len(self._pb_2_id)]] = neuron_id

    def id2source(self, neuron_id: Union[List[ID], ID]) \
            -> halco.PADIBusOnPADIBusBlock:
        """
        Get hardware coordinate from pyNN ID

        :param neuron_id: pyNN neuron ID
        """
        for padi_bus in self._pb_2_id:
            if all(i in self._pb_2_id[padi_bus] for i in neuron_id):
                return padi_bus
        raise RuntimeError("No ID found.")

    @staticmethod
    def _check_and_transform(lut: list) -> list:

        cell_id_size = BackgroundSpikeSourcePlacement._max_num_entries
        if len(lut) > cell_id_size:
            raise ValueError("Too many elements in HW LUT.")
        if len(lut) > len(set(lut)):
            raise ValueError("Non unique entries in HW LUT.")
        permutation = []
        for idx in lut:
            if not 0 <= idx < cell_id_size:
                raise ValueError(
                    f"BackgroundSpikeSourcePermutation list entry {idx} out of"
                    + f" range. Needs to be in range [0, {cell_id_size - 1}]"
                )
            coord = halco.PADIBusOnPADIBusBlock(idx)
            permutation.append(coord)
        return permutation


class State(BaseState):
    """Represent the simulator state."""

    # TODO support larger weights, ISSUE #3873
    max_weight: Final[int] = lola.SynapseMatrix.Weight.max

    # pylint: disable=invalid-name
    # TODO: replace by calculation (cf. feature #3594)
    dt: Final[float] = 3.4e-05  # average time between two MADC samples

    # pylint: disable=invalid-name
    def __init__(self):
        super(State, self).__init__()

        self.spikes = []
        self.times = []
        self.madc_samples = []

        self.mpi_rank = 0        # disabled
        self.num_processes = 1   # number of MPI processes
        self.running = False
        self.t = 0
        self.t_start = 0
        self.min_delay = 0
        self.max_delay = 0
        self.neuron_placement = None
        self.background_spike_source_placement = None
        self.populations: List[Population] = []
        self.recorders = set([])
        self.madc_recorder = None
        self.projections: List[Projection] = []
        self.plasticity_rules: List["PlasticityRule"] = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.log = logger.get("pyNN.brainscales2")
        self.injected_config = None
        self.injected_readout = None
        self.pre_realtime_tickets = None
        self.post_realtime_tickets = None
        self.pre_realtime_read = dict()
        self.post_realtime_read = dict()
        self.conn_manager = None
        self.conn = None
        self.conn_comes_from_outside = False
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_post_realtime = None
        self.initial_config = None

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.populations = []
        self.madc_recorder = None
        self.projections = []
        self.plasticity_rules = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.neuron_placement = None
        self.background_spike_source_placement = None
        self.injected_readout = None
        self.injected_config = None
        self.pre_realtime_tickets = None
        self.post_realtime_tickets = None
        self.pre_realtime_read = dict()
        self.post_realtime_read = dict()
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_post_realtime = None
        self.initial_config = None

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    @staticmethod
    def _get_spikes(network_graph: grenade.NetworkGraph,
                    outputs: grenade.IODataMap) -> Dict[int, np.ndarray]:
        """
        Get spikes indexed via neuron IDs.
        :param network_graph: Network graph to use for lookup of
                              spike label <-> ID relation
        :param outputs: All outputs of a single execution to extract
                        spikes from
        :return: Spikes as dict with atomic neuron enum value as key and
                 numpy array of times as value
        """
        spikes = grenade.extract_neuron_spikes(
            outputs, network_graph)
        if not spikes:
            return dict()
        assert len(spikes) == 1  # only one batch
        return spikes[0]

    @staticmethod
    def _get_v(network_graph: grenade.NetworkGraph,
               outputs: grenade.IODataMap) -> np.ndarray:
        """
        Get MADC samples with times in ms.
        :param network_graph: Network graph to use for lookup of
                              MADC output vertex descriptor
        :param outputs: All outputs of a single execution to extract
                        samples from
        :return: Times and sample values as numpy array
        """
        samples = grenade.extract_madc_samples(
            outputs, network_graph)
        if not samples:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
        assert len(samples) == 1  # only one batch
        return samples[0]

    # pylint: disable=too-many-arguments
    def _configure_hxneuron(self,
                            config: lola.Chip,
                            neuron_id: ID,
                            parameters: dict,
                            enable_spike_output: bool) \
            -> lola.Chip:
        """
        Places Neuron in Population "pop" on chip and configures spike and
        v recording.
        """

        # places the neurons from pop on chip
        atomic_neuron = HXNeuron.create_hw_entity(parameters)
        coord = self.neuron_placement.id2atomicneuron(neuron_id)

        # configure spike recording
        atomic_neuron.event_routing.analog_output = \
            atomic_neuron.EventRouting.AnalogOutputMode.normal
        atomic_neuron.event_routing.enable_digital = enable_spike_output

        config.neuron_block.atomic_neurons[coord] = atomic_neuron

        return config

    def _recorders_populations_changed(self) -> Set[Population]:
        """
        Collect populations which configurations were changed.

        This includes changes in:
            - neuron parameters
            - recorder settings
            - out-going synaptic connections

        :return: Populations which were subject to a change mentioned above.
        """
        changed = set()
        for recorder in self.recorders:
            population = recorder.population
            if (population.changed_since_last_run
                    or recorder.changed_since_last_run):
                changed.add(population)
        for projection in self.projections:
            pre_has_grandparent = hasattr(projection.pre, "grandparent")
            pre = projection.pre.grandparent if \
                pre_has_grandparent else projection.pre
            if projection.changed_since_last_run:
                changed.add(pre)
        return changed

    def _spike_source_indices(self) -> Dict[Population, Set[ID]]:
        """
        Collect all neurons which serve as a spike source.

        Check each projection and collect populations and their neurons which
        serve as spike sources.

        :return: Sets cell ids of neurons which serve as spike sources.
                 These sets are organized in populations which they belong to.
        """
        spike_source_indices = dict()
        for projection in self.projections:
            pre_has_grandparent = hasattr(projection.pre, "grandparent")
            pre = projection.pre.grandparent if \
                pre_has_grandparent else projection.pre
            for connection in projection.connections:
                if pre not in spike_source_indices:
                    spike_source_indices.update({pre: set()})
                spike_source_indices[pre].add(
                    pre.all_cells[connection.pop_pre_index])
        return spike_source_indices

    def _configure_recorders_populations(self,
                                         config: lola.Chip) \
            -> lola.Chip:

        changed = self._recorders_populations_changed()
        if not changed:
            return config
        spike_source_indices = self._spike_source_indices()
        for recorder in self.recorders:
            if recorder.population not in changed:
                continue
            population = recorder.population
            assert isinstance(population.celltype, (HXNeuron,
                                                    SpikeSourceArray,
                                                    SpikeSourcePoisson,
                                                    SpikeSourcePoissonOnChip))
            if isinstance(population.celltype, HXNeuron):

                # retrieve for which neurons what kind of recording is active
                spike_rec_indexes = set()
                for parameter, cell_ids in recorder.recorded.items():
                    for cell_id in cell_ids:
                        if parameter == "spikes":
                            spike_rec_indexes.add(cell_id)
                        elif parameter in recorder.madc_variables:
                            assert self.madc_recorder is not None and \
                                cell_id == self.madc_recorder.cell_id
                        else:
                            raise NotImplementedError
                if population in spike_source_indices:
                    spike_rec_indexes = spike_rec_indexes.union(
                        spike_source_indices[population])
                for cell_id, parameters in zip(
                        population.all_cells,
                        population.celltype.parameter_space):

                    config = self._configure_hxneuron(
                        config,
                        cell_id,
                        parameters,
                        enable_spike_output=cell_id in spike_rec_indexes)
        return config

    @staticmethod
    def _configure_routing(config: lola.Chip) -> lola.Chip:
        """
        Configure global routing-related but static parameters.
        :param config: Chip configuration to add configuration to
        :return: Altered chip configuration
        """

        # set synapse capmem cells
        for block in halco.iter_all(halco.SynapseBlockOnDLS):
            config.synapse_blocks[block].i_bias_dac.fill(1022)

        return config

    def _generate_network_graph(self) -> grenade.NetworkGraph:
        """
        Generate placed and routed executable network graph representation.
        """
        # check if populations, recorders, projections or plasticity changed
        changed_since_last_run = any(
            elem.changed_since_last_run for elem in itertools.chain(
                iter(self.populations),
                iter(self.recorders),
                iter(self.projections),
                iter(self.plasticity_rules)))
        if not changed_since_last_run:
            if self.grenade_network_graph is not None:
                return self.grenade_network_graph

        # generate network
        network_builder = grenade.NetworkBuilder()
        for pop in self.populations:
            pop.celltype.add_to_network_graph(
                pop, network_builder)
        for proj in self.projections:
            proj.add_to_network_graph(
                self.populations, proj, network_builder)
        for plasticity_rule in self.plasticity_rules:
            plasticity_rule.add_to_network_graph(network_builder)
        network = network_builder.done()

        # route network if required
        routing_result = None
        if self.grenade_network is None \
                or grenade.requires_routing(network, self.grenade_network):
            routing_result = grenade.build_routing(network)

        self.grenade_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_network_graph = grenade.build_network_graph(
                self.grenade_network, routing_result)
        else:
            grenade.update_network_graph(
                self.grenade_network_graph, self.grenade_network)

        return self.grenade_network_graph

    def _reset_changed_since_last_run(self):
        """
        Reset changed_since_last_run flag to track incremental changes for the
        next run.
        """
        for pop in self.populations:
            pop.changed_since_last_run = False
        for recorder in self.recorders:
            recorder.changed_since_last_run = False
        for proj in self.projections:
            proj.changed_since_last_run = False
        for plasticity_rule in self.plasticity_rules:
            plasticity_rule.changed_since_last_run = False

    def _generate_inputs(self, network_graph: grenade.NetworkGraph) \
            -> grenade.IODataMap:
        """
        Generate external input events from the routed network graph
        representation.
        """
        if network_graph.event_input_vertex is None:
            return grenade.IODataMap()
        input_generator = grenade.InputGenerator(network_graph)
        for population in self.populations:
            population.celltype.add_to_input_generator(
                population, input_generator)
        return input_generator.done()

    def _generate_playback_hooks(self):
        assert self.injection_pre_static_config is not None
        assert self.injection_pre_realtime is not None
        assert self.injection_post_realtime is not None
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()
        pre_static_config.copy_back(
            self.injection_pre_static_config)
        pre_realtime.copy_back(
            self.injection_pre_realtime)
        self._prepare_pre_realtime_read(pre_realtime)
        post_realtime.copy_back(
            self.injection_post_realtime)
        self._prepare_post_realtime_read(post_realtime)
        return grenade.ExecutionInstancePlaybackHooks(
            pre_static_config, pre_realtime, post_realtime)

    def _prepare_pre_realtime_read(self, builder: sta.PlaybackProgramBuilder):
        """
        Prepare injected readout after pre_realtime configuration and before
        realtime experiment section. This generates tickets to access the read
        information and ensures completion via a barrier.
        :param builder: Builder to append instructions to.
        """
        if not self.injected_readout.pre_realtime:
            return
        self.pre_realtime_tickets = {coord: builder.read(coord) for coord in
                                     self.injected_readout.pre_realtime}
        barrier = hal.Barrier()
        barrier.enable_omnibus = True
        barrier.enable_jtag = True
        builder.block_until(halco.BarrierOnFPGA(), barrier)

    def _prepare_post_realtime_read(self, builder: sta.PlaybackProgramBuilder):
        """
        Prepare injected readout after post_realtime configuration.
        This generates tickets to access the read information and ensures
        completion via a barrier.
        :param builder: Builder to append instructions to.
        """
        if not self.injected_readout.post_realtime:
            return
        self.post_realtime_tickets = {coord: builder.read(coord) for coord in
                                      self.injected_readout.post_realtime}
        barrier = hal.Barrier()
        barrier.enable_omnibus = True
        barrier.enable_jtag = True
        builder.block_until(halco.BarrierOnFPGA(), barrier)

    def _get_pre_realtime_read(self) -> Dict[halco.Coordinate, hal.Container]:
        """
        Redeem tickets of injected readout after pre_realtime section to get
        information after execution.
        :return: Dictionary with coordinates as keys and read container as
                 values.
        """
        if not self.pre_realtime_tickets:
            return dict()
        cocos = {coord: ticket.get() for coord, ticket in
                 self.pre_realtime_tickets.items()}
        return cocos

    def _get_post_realtime_read(self) -> Dict[halco.Coordinate, hal.Container]:
        """
        Redeem tickets of injected readout after post_realtime section to get
        information after execution.
        :return: Dictionary with coordinates as keys and read container as
                 values.
        """
        if not self.post_realtime_tickets:
            return dict()
        cocos = {coord: ticket.get() for coord, ticket in
                 self.post_realtime_tickets.items()}
        return cocos

    def prepare_static_config(self):
        if self.initial_config is None:
            config = lola.Chip()
        else:
            config = copy(self.initial_config)
        builder1 = sta.PlaybackProgramBuilder()

        # generate common static configuration
        config = self._configure_routing(config)

        def add_configuration(
                builder: sta.PlaybackProgramBuilder,
                additional_configuration: Union[
                    Dict[halco.Coordinate, hal.Container],
                    sta.PlaybackProgramBuilder]):
            if isinstance(additional_configuration,
                          sta.PlaybackProgramBuilder):
                builder.merge_back(additional_configuration)
            else:
                tmpdumper = sta.DumperDone()
                tmpdumper.values = list(additional_configuration.items())
                builder.merge_back(sta.convert_to_builder(tmpdumper))

        # injected configuration pre non realtime
        add_configuration(builder1, self.injected_config.pre_non_realtime)

        self.grenade_chip_config = config
        # reset dirty-flags
        self._reset_changed_since_last_run()

        # injected configuration pre realtime
        pre_realtime = sta.PlaybackProgramBuilder()
        add_configuration(pre_realtime, self.injected_config.pre_realtime)

        # injected configuration post realtime
        post_realtime = sta.PlaybackProgramBuilder()
        add_configuration(post_realtime, self.injected_config.post_realtime)

        self.injection_pre_static_config = builder1
        self.injection_pre_realtime = pre_realtime
        self.injection_post_realtime = post_realtime

    def run(self, runtime: Optional[float]):
        """
        Performs a hardware run for `runtime` milliseconds.
        If runtime is `None`, we only perform preparatory steps.
        """
        time_begin = time.time()
        if runtime is None:
            self.log.INFO("User requested 'None' runtime: "
                          + "no hardware run performed.")
        else:
            self.t += runtime
        self.running = True

        # generate network graph
        network_graph = self._generate_network_graph()

        # configure populations and recorders
        self.grenade_chip_config = self._configure_recorders_populations(
            self.grenade_chip_config)

        self._reset_changed_since_last_run()

        if runtime is None:
            self.log.DEBUG("run(): Completed in {:.3f}s".format(
                time.time() - time_begin))
            return

        # generate external spike trains
        inputs = self._generate_inputs(network_graph)
        runtime_in_clocks = int(
            runtime * int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        if runtime_in_clocks > hal.Timer.Value.max:
            max_runtime = hal.Timer.Value.max /\
                1000 / int(hal.Timer.Value.fpga_clock_cycles_per_us)
            raise ValueError(f"Runtime of {runtime} to long. "
                             f"Maximum supported runtime {max_runtime}")

        inputs.runtime = [runtime_in_clocks]

        if not self.conn_comes_from_outside and \
           self.conn_manager is None:
            self.conn_manager = grenade.ManagedConnection()
            assert self.conn is None
            self.conn = self.conn_manager.__enter__()

        time_after_preparations = time.time()
        self.log.DEBUG("run(): Preparations finished in {:.3f}s".format(
            time_after_preparations - time_begin))

        outputs = grenade.run(
            self.conn, self.grenade_chip_config, network_graph,
            inputs, self._generate_playback_hooks())

        self.log.DEBUG("run(): Execution finished in {:.3f}s".format(
            time.time() - time_after_preparations))
        time_after_hw_run = time.time()

        # make list 'spikes' of tupel (neuron id, spike time)
        self.spikes = self._get_spikes(network_graph, outputs)

        # make two list for madc samples: times, madc_samples
        self.times, self.madc_samples = self._get_v(
            network_graph, outputs)

        self.pre_realtime_read = self._get_pre_realtime_read()
        self.post_realtime_read = self._get_post_realtime_read()

        self.log.DEBUG("run(): Postprocessing finished in {:.3f}s".format(
            time.time() - time_after_hw_run))
        self.log.DEBUG("run(): Completed in {:.3f}s".format(
            time.time() - time_begin))


# state is instantiated in setup()
state: Optional[State] = None
