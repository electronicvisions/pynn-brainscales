import itertools
from typing import Optional, Final, List, Dict, Union
import numpy as np
from pyNN.common import IDMixin, Population, Projection
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells import HXNeuron, \
    SpikeSourceArray, SpikeSourcePoisson
from dlens_vx_v2 import hal, halco, sta, hxcomm, lola, logger
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
        self.populations: List[Population] = []
        self.recorders = set([])
        self.madc_recorder = None
        self.projections: List[Projection] = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.enable_neuron_bypass = False
        self.log = logger.get("pyNN.brainscales2")
        self.injected_config = None
        self.conn_manager = None
        self.conn = None
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_post_realtime = None

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.populations = []
        self.madc_recorder = None
        self.projections = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.enable_neuron_bypass = False
        self.neuron_placement = None
        self.injected_config = None
        self.grenade_network = None
        self.grenade_network_graph = None
        self.grenade_chip_config = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_post_realtime = None

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    @staticmethod
    def _get_spikes(network_graph: grenade.NetworkGraph,
                    outputs: grenade.IODataMap) -> np.ndarray:
        """
        Get spikes indexed via neuron IDs.
        :param network_graph: Network graph to use for lookup of
                              spike label <-> ID relation
        :param outputs: All outputs of a single execution to extract
                        spikes from
        :return: Spikes as (time, ID) as numpy array
        """
        times, neurons = grenade.extract_neuron_spikes(
            outputs, network_graph)
        return np.array((neurons, times)).T

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
        times, samples = grenade.extract_madc_samples(
            outputs, network_graph)
        return times, samples

    @staticmethod
    def _configure_common(builder: sta.PlaybackProgramBuilder,
                          config: grenade.ChipConfig) \
            -> (sta.PlaybackProgramBuilder, grenade.ChipConfig):

        # set global cells
        neuron_params = {
            halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n: 340,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp: 110,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_leak_source_follower:
            100,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_spike_comparator:
            500}

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for key, value in neuron_params.items():
                builder.write(halco.CapMemCellOnDLS(key, block),
                              hal.CapMemCell(value))

        # set all neurons on chip to default values
        default_neuron = HXNeuron.create_hw_entity({})
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            config.hemispheres[coord.toNeuronRowOnDLS().toHemisphereOnDLS()]\
                .neuron_block[coord.toNeuronColumnOnDLS()] = default_neuron

        neuron_config = hal.CommonNeuronBackendConfig()
        neuron_config.clock_scale_fast = 3
        neuron_config.clock_scale_slow = 3
        neuron_config.enable_clocks = True
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            config.neuron_backend[coord] = neuron_config

        return builder, config

    # pylint: disable=too-many-arguments
    def _configure_hxneuron(self,
                            config: grenade.ChipConfig,
                            neuron_id: ID,
                            parameters: dict,
                            readout_source:
                            Optional[hal.NeuronConfig.ReadoutSource]) \
            -> grenade.ChipConfig:
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
        atomic_neuron.event_routing.enable_digital = True
        if self.enable_neuron_bypass:
            # disable threshold comparator
            atomic_neuron.threshold.enable = False
            atomic_neuron.event_routing.enable_bypass_excitatory = True
            atomic_neuron.event_routing.enable_bypass_inhibitory = True

        # configure v recording
        if readout_source is not None:
            atomic_neuron.readout.enable_amplifier = True
            atomic_neuron.readout.enable_buffered_access = True
            atomic_neuron.readout.source = readout_source

        config.hemispheres[coord.toNeuronRowOnDLS().toHemisphereOnDLS()]\
            .neuron_block[coord.toNeuronColumnOnDLS()] = atomic_neuron

        return config

    def _configure_recorders_populations(self,
                                         config: grenade.ChipConfig) \
            -> grenade.ChipConfig:

        for recorder in self.recorders:
            population = recorder.population
            if not (population.changed_since_last_run
                    or recorder.changed_since_last_run):
                continue
            assert isinstance(population.celltype, (HXNeuron,
                                                    SpikeSourceArray,
                                                    SpikeSourcePoisson))
            if isinstance(population.celltype, HXNeuron):

                # retrieve for which neurons what kind of recording is active
                spike_rec_indexes = []
                madc_recording_id = None
                readout_source = Optional[hal.NeuronConfig.ReadoutSource]
                for parameter, cell_ids in recorder.recorded.items():
                    for cell_id in cell_ids:
                        # we always record spikes at the moment
                        spike_rec_indexes.append(cell_id)
                        if parameter == "spikes":
                            pass
                        elif parameter in recorder.madc_variables:
                            assert self.madc_recorder is not None and \
                                cell_id == self.madc_recorder.cell_id
                            madc_recording_id = cell_id
                            readout_source = self.madc_recorder.readout_source
                        else:
                            raise NotImplementedError
                for cell_id, parameters in zip(
                        population.all_cells,
                        population.celltype.parameter_space):

                    this_source = None
                    if cell_id == madc_recording_id:
                        this_source = readout_source
                    config = self._configure_hxneuron(
                        config,
                        cell_id,
                        parameters,
                        readout_source=this_source)
        return config

    @staticmethod
    def _configure_routing(builder: sta.PlaybackProgramBuilder,
                           config: grenade.ChipConfig) \
            -> (sta.PlaybackProgramBuilder, grenade.ChipConfig):
        """
        Configure global routing-related but static parameters.
        :param builder: Playback program builder to add configuration to
        :config: Chip configuration to add configuration to
        :return: Altered playback program builder and chip configuration
        """

        # configure PADI bus
        padi_config = hal.CommonPADIBusConfig()
        for block in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            if state.enable_neuron_bypass:
                # extend pulse length such that pre-synaptic signals have
                # a stronger effect on the synaptic input voltage and spikes
                # are more easily detected by the bypass circuit.
                # pylint: disable=unsupported-assignment-operation
                padi_config.dacen_pulse_extension[block] = \
                    hal.CommonPADIBusConfig.DacenPulseExtension.max
        for padibus in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            config.hemispheres[padibus.toHemisphereOnDLS()]\
                .common_padi_bus_config = padi_config

        # configure switches
        # TODO (Issue #3745, #3746): move to AtomicNeuron and set in `grenade`
        current_switch_quad = hal.ColumnCurrentQuad()
        switch = current_switch_quad.ColumnCurrentSwitch()
        switch.enable_synaptic_current_excitatory = True
        switch.enable_synaptic_current_inhibitory = True

        for s in halco.iter_all(halco.EntryOnQuad):
            current_switch_quad.set_switch(s, switch)

        for sq in halco.iter_all(halco.ColumnCurrentQuadOnDLS):
            builder.write(sq, current_switch_quad)

        # set synapse capmem cells
        synapse_params = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1022,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 1010,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 1010,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 1010}
        if state.enable_neuron_bypass:
            synapse_params[halco.CapMemCellOnCapMemBlock.syn_i_bias_dac] = 1022

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in synapse_params.items():
                builder.write(halco.CapMemCellOnDLS(k, block),
                              hal.CapMemCell(v))

        return builder, config

    def _perform_post_fail_analysis(self, connection):
        """
        Read out and log FPGA status containers in a post-mortem program.
        """
        builder = sta.PlaybackProgramBuilder()

        # perform stat readout at the end of the experiment
        ticket_arq = builder.read(halco.HicannARQStatusOnFPGA())

        tickets_phy = []
        for coord in halco.iter_all(halco.PhyStatusOnFPGA):
            tickets_phy.append(builder.read(coord))

        builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
        sta.run(connection, builder.done())

        error_msg = "_perform_post_fail_analysis(): "
        error_msg += "Experiment failed, reading post-mortem status."
        error_msg += str(ticket_arq.get())
        for ticket_phy in tickets_phy:
            error_msg += str(ticket_phy.get()) + "\n"
        self.log.ERROR(error_msg)

    def _generate_network_graph(self) -> grenade.NetworkGraph:
        """
        Generate placed and routed executable network graph representation.
        """
        # check if populations, recorders or projections changed
        changed_since_last_run = any(
            elem.changed_since_last_run for elem in itertools.chain(
                iter(self.populations),
                iter(self.recorders),
                iter(self.projections)))
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
        post_realtime.copy_back(
            self.injection_post_realtime)
        return grenade.ExecutionInstancePlaybackHooks(
            pre_static_config, pre_realtime, post_realtime)

    def prepare_static_config(self):
        config = grenade.ChipConfig()
        builder1 = sta.PlaybackProgramBuilder()

        # generate common static configuration
        builder1, config = self._configure_common(builder1, config)
        builder1, config = self._configure_routing(builder1, config)

        # injected configuration pre non realtime
        tmpdumper = sta.DumperDone()
        tmpdumper.values = list(self.injected_config.pre_non_realtime.items())
        config = grenade.convert_to_chip(tmpdumper, config)
        builder1.merge_back(sta.convert_to_builder(tmpdumper))

        self.grenade_chip_config = config

        # reset dirty-flags
        self._reset_changed_since_last_run()

        # injected configuration post non realtime
        tmpdumper = sta.DumperDone()
        tmpdumper.values = list(self.injected_config.post_non_realtime.items())
        pre_realtime = sta.convert_to_builder(tmpdumper)

        # injected configuration pre realtime
        tmpdumper = sta.DumperDone()
        tmpdumper.values = list(self.injected_config.pre_realtime
                                .items())
        pre_realtime.merge_back(
            sta.convert_to_builder(tmpdumper))

        # injected configuration post realtime
        tmpdumper = sta.DumperDone()
        tmpdumper.values = list(self.injected_config.post_realtime
                                .items())
        post_realtime = sta.convert_to_builder(tmpdumper)

        self.injection_pre_static_config = builder1
        self.injection_pre_realtime = pre_realtime
        self.injection_post_realtime = post_realtime

    def run(self, runtime: Optional[float]):
        """
        Performs a hardware run for `runtime` milliseconds.
        If runtime is `None`, we only perform preparatory steps.
        """
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
            return

        # generate external spike trains
        inputs = self._generate_inputs(network_graph)
        inputs.runtime = \
            [int(runtime
                 * int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)]

        if self.conn_manager is None:
            self.conn_manager = hxcomm.ManagedConnection()
            assert self.conn is None
            self.conn = self.conn_manager.__enter__()

        try:
            outputs = grenade.run(
                self.conn, self.grenade_chip_config, network_graph,
                inputs, self._generate_playback_hooks())

        except RuntimeError:
            # perform post-mortem read out of status
            self._perform_post_fail_analysis(self.conn)
            raise

        # make list 'spikes' of tupel (neuron id, spike time)
        self.spikes = self._get_spikes(network_graph, outputs)

        # make two list for madc samples: times, madc_samples
        self.times, self.madc_samples = self._get_v(
            network_graph, outputs)


# state is instantiated in setup()
state: Optional[State] = None
