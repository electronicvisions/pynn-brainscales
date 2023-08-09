import time
import itertools
from copy import copy
from typing import Optional, Final, List, Dict, Union, Tuple, NamedTuple, Any
import numpy as np
from pyNN.common import IDMixin, Population, Projection
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells_base import \
    StandardCellType
from dlens_vx_v3 import hal, halco, sta, lola, logger
import pygrenade_vx as grenade
from calix.spiking import SpikingCalibTarget, SpikingCalibOptions
from calix.spiking.neuron import NeuronCalibTarget
from calix import calibrate


name = "HX"  # for use in annotating output data


class MADCRecording(NamedTuple):
    '''
    Times and values of a MADC recording.
    '''
    times: np.ndarray
    values: np.ndarray


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        int.__init__(n)
        IDMixin.__init__(self)


class NeuronPlacement:
    """
    Tracks the assignment of pyNN IDs to LogicalNeuronOnDLS.

    This tracking is needed for all neuron types which are placed in the
    neuron array. By default the anchor of the neurons are placed in increasing
    order of the hardware enumeration.

    :param neuron_id: Look up table for permutation. Index: HW related
                    population neuron enumeration. Value: HW neuron
                    enumeration.
    """
    _id_2_coord: Dict[ID, halco.LogicalNeuronOnDLS]
    _available_coords: np.ndarray
    _MAX_NUM_ENTRIES: Final[int] = halco.AtomicNeuronOnDLS.size
    DEFAULT_PERMUTATION: Final[List[int]] = range(halco.AtomicNeuronOnDLS.size)

    def __init__(self, permutation: List[int] = None):
        if permutation is None:
            permutation = range(self._MAX_NUM_ENTRIES)
        self._id_2_coord = {}
        self._available_coords = self._check_and_transform(permutation)

    def register_neuron(self, neuron_id: Union[List[ID], ID],
                        logical_compartments: halco.LogicalNeuronCompartments):
        """
        Register new IDs to placement.

        :param neuron_id: pyNN neuron IDs to be registered.
        :param logical_compartments: LogicalNeuronCompartments which belong
            to the neurons which should be registered. All neurons which should
            be registered have to share the same morphology, i.e. have the same
            LogicalNeuronCompartments coordinate.
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]

        compartments = logical_compartments.get_compartments().values()
        circuits_per_neuron = np.sum([len(comp) for comp in compartments])
        if len(neuron_id) * circuits_per_neuron > len(self._available_coords):
            raise ValueError(
                f"Cannot register more than {len(self._available_coords)} "
                "neuron circuits.")
        for idx in neuron_id:
            placed = False
            # try one available coordinate after another as an anchor
            for anchor in self._available_coords:
                try:
                    logical_neuron = halco.LogicalNeuronOnDLS(
                        logical_compartments, anchor)
                except RuntimeError:
                    # logical neuron extends over edge of neuron array
                    continue

                atomic_neurons = logical_neuron.get_atomic_neurons()
                if not np.isin(atomic_neurons, self._available_coords).all():
                    continue

                self._available_coords = self._available_coords[
                    ~np.isin(self._available_coords, atomic_neurons)]
                self._id_2_coord[idx] = logical_neuron
                placed = True
                break

            if not placed:
                raise ValueError("LogicalNeuron cannot be placed.")

    def id2logicalneuron(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[halco.LogicalNeuronOnDLS], halco.LogicalNeuronOnDLS]:
        """
        Get hardware coordinate from pyNN ID
        :param neuron_id: pyNN neuron ID
        """
        try:
            return [self._id_2_coord[idx] for idx in
                    neuron_id]
        except TypeError:
            return self._id_2_coord[neuron_id]

    def id2first_circuit(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[int], int]:
        """
        Get hardware coordinate of first circuit in first compartment as plain
        int from pyNN ID.

        :param neuron_id: pyNN neuron ID
        :return: Enums of first circuits in first compartments.
        """
        logical_neurons = self.id2logicalneuron(neuron_id)
        try:
            return [int(idx.get_atomic_neurons()[0].toEnum()) for
                    idx in logical_neurons]
        except TypeError:
            return int(logical_neurons.get_atomic_neurons()[0].toEnum())

    @staticmethod
    def _check_and_transform(lut: list) -> list:

        cell_id_size = NeuronPlacement._MAX_NUM_ENTRIES
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
        return np.array(permutation)


class BackgroundSpikeSourcePlacement:
    """
    Tracks assignment of pyNN IDs of SpikeSourcePoissonOnChip based populations
    to the corresponding hardware entity, i.e. BackgroundSpikeSourceOnDLS. We
    use one source on each hemisphere to ensure arbitrary routing works.
    Default constructed with reversed 1 to 1 permutation to yield better
    distribution for small networks.

    :cvar DEFAULT_PERMUTATION: Default permutation, where allocation is ordered
                               to start at the highest-enum PADI-bus to reduce
                               overlap with allocated neurons.
    """
    _pb_2_id: Dict[halco.PADIBusOnPADIBusBlock, List[ID]]
    _permutation: List[halco.PADIBusOnPADIBusBlock]
    _MAX_NUM_ENTRIES: Final[int] = halco.PADIBusOnPADIBusBlock.size
    DEFAULT_PERMUTATION: Final[List[int]] = list(reversed(range(
        halco.PADIBusOnPADIBusBlock.size)))

    def __init__(self, permutation: List[int] = None):
        """
        :param permutation: Look up table for permutation. Index: HW related
                            population neuron enumeration. Value: HW neuron
                            enumeration.
        """

        if permutation is None:
            permutation = self.DEFAULT_PERMUTATION
        self._pb_2_id = {}
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
        for padi_bus, ids in self._pb_2_id.items():
            if all(i in ids for i in neuron_id):
                return padi_bus
        raise RuntimeError("No ID found.")

    @staticmethod
    def _check_and_transform(lut: list) -> list:

        cell_id_size = BackgroundSpikeSourcePlacement._MAX_NUM_ENTRIES
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

    # pylint: disable=invalid-name
    # TODO: replace by calculation (cf. feature #3594)
    dt: Final[float] = 3.4e-05  # average time between two MADC samples

    # pylint: disable=invalid-name,too-many-statements
    def __init__(self):
        super().__init__()

        # BSS hardware can only record IrregularlySampledSignal
        self.record_sample_times = True
        self.spikes: Dict[Tuple[
            grenade.network.PopulationDescriptor, int,
            halco.CompartmentOnLogicalNeuron], List[float]] = {}
        self.madc_recordings = {}

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
        self.madc_recording_sites = {}
        self.projections: List[Projection] = []
        self.plasticity_rules: List["PlasticityRule"] = []
        self.synaptic_observables: List[Dict[str, object]] = []
        self.neuronal_observables: List[Dict[str, object]] = []
        self.array_observables: List[Dict[str, object]] = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.log = logger.get("pyNN.brainscales2")
        self.injected_config = None
        self.injected_readout = None
        self.pre_realtime_tickets = None
        self.inside_realtime_begin_tickets = None
        self.inside_realtime_end_tickets = None
        self.post_realtime_tickets = None
        self.pre_realtime_read = {}
        self.inside_realtime_begin_read = {}
        self.inside_realtime_end_read = {}
        self.post_realtime_read = {}
        self.conn_manager = None
        self.conn = None
        self.conn_comes_from_outside = False
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_inside_realtime_begin = None
        self.injection_inside_realtime_end = None
        self.injection_post_realtime = None
        self.initial_config = None
        self.execution_time_info = None
        self.calib_cache_dir = None

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.populations = []
        self.madc_recording_sites = {}
        self.projections = []
        self.plasticity_rules = []
        self.synaptic_observables = []
        self.neuronal_observables = []
        self.array_observables = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.neuron_placement = None
        self.background_spike_source_placement = None
        self.injected_readout = None
        self.injected_config = None
        self.pre_realtime_tickets = None
        self.inside_realtime_begin_tickets = None
        self.inside_realtime_end_tickets = None
        self.post_realtime_tickets = None
        self.pre_realtime_read = {}
        self.inside_realtime_begin_read = {}
        self.inside_realtime_end_read = {}
        self.post_realtime_read = {}
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_inside_realtime_begin = None
        self.injection_inside_realtime_end = None
        self.injection_post_realtime = None
        self.initial_config = None
        self.execution_time_info = None
        self.calib_cache_dir = None

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    def _get_v(self,
               network_graph: grenade.network.NetworkGraph,
               outputs: grenade.signal_flow.IODataMap
               # Note: Any should be recording.MADCRecordingSite. We do not
               # annotate the correct type due to cyclic imports.
               ) -> Dict[Any, MADCRecording]:
        """
        Get MADC samples with times in ms.
        :param network_graph: Network graph to use for lookup of
                              MADC output vertex descriptor
        :param outputs: All outputs of a single execution to extract
                        samples from
        :return: Dictionary with a madc recording site as key and a
            MADCRecording as value.
        """
        samples = grenade.network.extract_madc_samples(
            outputs, network_graph)[0]

        data = {}
        for site in self.madc_recording_sites:
            local_times, population, neuron_on_population, \
                compartment_on_neuron, _, local_values = samples
            # converting compartment_on_neuron to an integer increases the
            # speed of the comparison
            local_filter = (population == site.population) \
                & (neuron_on_population == site.neuron_on_population) \
                & (compartment_on_neuron == int(site.compartment_on_neuron))
            data[site] = MADCRecording(times=local_times[local_filter],
                                       values=local_values[local_filter])
        return data

    def _get_synaptic_observables(
            self,
            network_graph: grenade.network.NetworkGraph,
            outputs: grenade.signal_flow.IODataMap
    ) -> List[Dict[str, np.ndarray]]:
        """
        Get synaptic observables.
        :param network_graph: Network graph to use for lookup of
                              plasticity rule descriptor.
        :param outputs: All outputs of a single execution to extract
                        samples from.
        :return: List over projections and recorded data.
        """

        observables = [{} for projection in self.projections]
        for plasticity_rule in self.plasticity_rules:
            if not plasticity_rule.observables:
                continue
            data = plasticity_rule.get_data(network_graph, outputs)
            for obsv_name, data in data.data_per_synapse.items():
                for descriptor, value in data.items():
                    observables[descriptor].update({obsv_name: value[0]})
        return observables

    def _get_neuronal_observables(
            self,
            network_graph: grenade.network.NetworkGraph,
            outputs: grenade.signal_flow.IODataMap) -> Dict[str, np.ndarray]:
        """
        Get neuronal observables.
        :param network_graph: Network graph to use for lookup of
                              plasticity rule descriptor
        :param outputs: All outputs of a single execution to extract
                        samples from
        :return: Dict over projections and recorded data
        """

        observables = [{} for population in self.populations]
        for plasticity_rule in self.plasticity_rules:
            if not plasticity_rule.observables:
                continue
            data = plasticity_rule.get_data(network_graph, outputs)
            for obsv_name, data in data.data_per_neuron.items():
                for descriptor, value in data.items():
                    observables[descriptor].update({obsv_name: value[0]})
        return observables

    def _get_array_observables(
            self,
            network_graph: grenade.network.NetworkGraph,
            outputs: grenade.signal_flow.IODataMap) \
            -> List[Dict[str, np.ndarray]]:
        """
        Get general array observables.

        :param network_graph: Network graph to use for lookup of
                              plasticity rule descriptor
        :param outputs: All outputs of a single execution to extract
                        samples from
        :return: List of dicts over plasticity rules and recorded data,
            one dict per plasticity rule
        """

        observables = []
        for plasticity_rule in self.plasticity_rules:
            if not plasticity_rule.observables:
                observables.append({})
            else:
                data = plasticity_rule.get_data(network_graph, outputs)
                observables.append(data.data_array)
        return observables

    def _configure_recorders_populations(self):
        changed = set()
        for population in self.populations:
            if population.changed_since_last_run:
                changed.add(population)
        if not changed:
            return

        neuron_target = NeuronCalibTarget().DenseDefault
        # initialize shared parameters between neurons with None to allow check
        # for different values in different populations
        neuron_target.synapse_dac_bias = None
        neuron_target.i_synin_gm = np.array([None, None])

        # gather calibration information
        execute_calib = False
        for population in changed:
            assert isinstance(population.celltype, StandardCellType)
            if hasattr(population.celltype, 'add_calib_params'):
                population.celltype.add_calib_params(
                    neuron_target, population.all_cells)
                execute_calib = True

        if execute_calib:
            if self.initial_config is not None:
                self.log.WARN("Using automatically calibrating neurons with "
                              "initial_config. Initial configuration will be "
                              "overwritten")
            calib_target = SpikingCalibTarget(neuron_target=neuron_target)
            # release JITGraphExecuter connection to establish a new one for
            # calibration (JITGraphExecuter conenctions can not be shared with
            # lower layers).
            if self.conn is not None and not self.conn_comes_from_outside:
                self.conn_manager.__exit__()
            result = calibrate(
                calib_target,
                SpikingCalibOptions(),
                self.calib_cache_dir)
            if self.conn is not None and not self.conn_comes_from_outside:
                self.conn = self.conn_manager.__enter__()
            dumper = sta.PlaybackProgramBuilderDumper()
            result.apply(dumper)
            self.grenade_chip_config = sta.convert_to_chip(
                dumper.done(), self.grenade_chip_config)

        for population in changed:
            if hasattr(population.celltype, 'add_to_chip'):
                population.celltype.add_to_chip(
                    population.all_cells, self.grenade_chip_config)

    def _generate_network_graph(self):
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
                return

        # generate network
        network_builder = grenade.network.NetworkBuilder()
        for pop in self.populations:
            pop.celltype.add_to_network_graph(
                pop, network_builder)
        for proj in self.projections:
            proj.add_to_network_graph(
                self.populations, proj, network_builder)
        for plasticity_rule in self.plasticity_rules:
            plasticity_rule.add_to_network_graph(network_builder)
        # generate MADC recording
        if len(self.madc_recording_sites) > 0:
            assert len(self.madc_recording_sites) <= 2
            madc_recording_neurons = []
            for rec_site, source in self.madc_recording_sites.items():
                neuron = grenade.network.MADCRecording.Neuron()
                neuron.coordinate.population = grenade.network\
                    .PopulationDescriptor(rec_site.population)
                neuron.source = source
                neuron.coordinate.neuron_on_population \
                    = rec_site.neuron_on_population
                neuron.coordinate.compartment_on_neuron \
                    = rec_site.compartment_on_neuron
                neuron.coordinate.atomic_neuron_on_compartment = 0
                madc_recording_neurons.append(neuron)
            madc_recording = grenade.network.MADCRecording(
                madc_recording_neurons)
            network_builder.add(madc_recording)

        network = network_builder.done()

        # route network if required
        routing_result = None
        if self.grenade_network_graph is None \
                or grenade.network.requires_routing(
                    network, self.grenade_network_graph):
            routing_result = grenade.network.routing.PortfolioRouter()(
                network)

        self.grenade_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_network_graph = grenade.network\
                .build_network_graph(
                    self.grenade_network, routing_result)
        else:
            grenade.network.update_network_graph(
                self.grenade_network_graph, self.grenade_network)

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

    def _generate_inputs(
            self, network_graph: grenade.network.NetworkGraph) \
            -> grenade.signal_flow.IODataMap:
        """
        Generate external input events from the routed network graph
        representation.
        """
        if network_graph.event_input_vertex is None:
            return grenade.signal_flow.IODataMap()
        input_generator = grenade.network.InputGenerator(
            network_graph)
        for population in self.populations:
            population.celltype.add_to_input_generator(
                population, input_generator)
        return input_generator.done()

    def _generate_playback_hooks(self):
        assert self.injection_pre_static_config is not None
        assert self.injection_pre_realtime is not None
        assert self.injection_inside_realtime_begin is not None
        assert self.injection_inside_realtime_end is not None
        assert self.injection_post_realtime is not None
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()
        pre_static_config.copy_back(
            self.injection_pre_static_config)
        pre_realtime.copy_back(
            self.injection_pre_realtime)
        self._prepare_pre_realtime_read(pre_realtime)
        inside_realtime_begin.copy_back(
            self.injection_inside_realtime_begin)
        self._prepare_inside_realtime_begin_read(inside_realtime_begin)
        inside_realtime_end.copy_back(
            self.injection_inside_realtime_end)
        self._prepare_inside_realtime_end_read(inside_realtime_end)
        post_realtime.copy_back(
            self.injection_post_realtime)
        self._prepare_post_realtime_read(post_realtime)
        return grenade.signal_flow.ExecutionInstancePlaybackHooks(
            pre_static_config, pre_realtime, inside_realtime_begin,
            inside_realtime_end, post_realtime)

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

    def _prepare_inside_realtime_begin_read(
            self, builder: sta.PlaybackProgramBuilder):
        """
        Prepare injected readout after inside_realtime_begin configuration and
        before event insertion. This generates tickets to access the read
        information and ensures completion via a barrier.
        :param builder: Builder to append instructions to.
        """
        if not self.injected_readout.inside_realtime_begin:
            return
        self.inside_realtime_begin_tickets = {
            coord: builder.read(coord) for
            coord in self.injected_readout.inside_realtime_begin}
        barrier = hal.Barrier()
        barrier.enable_omnibus = True
        barrier.enable_jtag = True
        builder.block_until(halco.BarrierOnFPGA(), barrier)

    def _prepare_inside_realtime_end_read(
            self, builder: sta.PlaybackProgramBuilder):
        """
        Prepare injected readout after inside_realtime_end configuration and
        after realtime_end experiment section. This generates tickets to access
        the read information and ensures completion via a barrier.
        :param builder: Builder to append instructions to.
        """
        if not self.injected_readout.inside_realtime_end:
            return
        self.inside_realtime_end_tickets = {
            coord: builder.read(coord) for coord in
            self.injected_readout.inside_realtime_end}
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
            return {}
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
            return {}
        cocos = {coord: ticket.get() for coord, ticket in
                 self.post_realtime_tickets.items()}
        return cocos

    def prepare_static_config(self):
        if self.initial_config is None:
            self.grenade_chip_config = lola.Chip()
        else:
            self.grenade_chip_config = copy(self.initial_config)
        builder1 = sta.PlaybackProgramBuilder()

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

        # reset dirty-flags
        self._reset_changed_since_last_run()

        # injected configuration pre realtime
        pre_realtime = sta.PlaybackProgramBuilder()
        add_configuration(pre_realtime, self.injected_config.pre_realtime)

        # injected configuration inside realtime begin
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        add_configuration(inside_realtime_begin,
                          self.injected_config.inside_realtime_begin)

        # injected configuration inside realtime_end end
        inside_realtime_end = sta.PlaybackProgramBuilder()
        add_configuration(inside_realtime_end,
                          self.injected_config.inside_realtime_end)

        # injected configuration post realtime
        post_realtime = sta.PlaybackProgramBuilder()
        add_configuration(post_realtime, self.injected_config.post_realtime)

        self.injection_pre_static_config = builder1
        self.injection_pre_realtime = pre_realtime
        self.injection_inside_realtime_begin = inside_realtime_begin
        self.injection_inside_realtime_end = inside_realtime_end
        self.injection_post_realtime = post_realtime

    def preprocess(self):
        """
        Execute all steps needed for the hardware back-end.
        Includes place&route of network graph or execution of calibration.
        Can be called manually to obtain calibration results for e.g.
        CalibHXNeuronCuba/Coba and make adjustments if needed.
        If not called manually is automatically called on run().
        """
        self._generate_network_graph()
        self._configure_recorders_populations()
        self._reset_changed_since_last_run()

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

        self.preprocess()

        if runtime is None:
            self.log.DEBUG(
                f"run(): Completed in {(time.time() - time_begin):.3f}s")
            return

        # generate external spike trains
        inputs = self._generate_inputs(self.grenade_network_graph)
        runtime_in_clocks = int(
            runtime * int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        if runtime_in_clocks > hal.Timer.Value.max:
            max_runtime = hal.Timer.Value.max /\
                1000 / int(hal.Timer.Value.fpga_clock_cycles_per_us)
            raise ValueError(f"Runtime of {runtime} to long. "
                             f"Maximum supported runtime {max_runtime}")

        inputs.runtime = [{grenade.common.ExecutionInstanceID():
                           runtime_in_clocks}]

        if not self.conn_comes_from_outside and \
           self.conn_manager is None:
            self.conn_manager = grenade.execution.ManagedJITGraphExecutor()
            assert self.conn is None
            self.conn = self.conn_manager.__enter__()

        time_after_preparations = time.time()
        self.log.DEBUG("run(): Preparations finished in "
                       f"{(time_after_preparations - time_begin):.3f}s")

        outputs = grenade.network.run(
            self.conn, self.grenade_chip_config, self.grenade_network_graph,
            inputs, self._generate_playback_hooks())

        self.log.DEBUG("run(): Execution finished in "
                       f"{(time.time() - time_after_preparations):.3f}s")
        time_after_hw_run = time.time()

        self.spikes = grenade.network.extract_neuron_spikes(
            outputs, self.grenade_network_graph)[0]

        self.madc_recordings = self._get_v(self.grenade_network_graph, outputs)

        self.synaptic_observables = self._get_synaptic_observables(
            self.grenade_network_graph, outputs)
        self.array_observables = self._get_array_observables(
            self.grenade_network_graph, outputs)

        self.neuronal_observables = self._get_neuronal_observables(
            self.grenade_network_graph, outputs)

        self.pre_realtime_read = self._get_pre_realtime_read()
        self.post_realtime_read = self._get_post_realtime_read()

        self.execution_time_info = outputs.execution_time_info
        assert self.execution_time_info is not None

        self.log.DEBUG("run(): Postprocessing finished in "
                       f"{(time.time() - time_after_hw_run):.3f}s")
        self.log.DEBUG("run(): Completed in "
                       f"{(time.time() - time_begin):.3f}s")


# state is instantiated in setup()
state: Optional[State] = None
