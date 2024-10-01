import time
from copy import deepcopy
from typing import Optional, Final, List, Dict, Union, NamedTuple
import numpy as np
from pyNN.common import IDMixin, Population, Projection
from pyNN.common.control import BaseState
from dlens_vx_v3 import hal, halco, sta, logger
import pygrenade_vx as grenade
import pygrenade_common as grenade_common

from pynn_brainscales.brainscales2.recording_data import Recording


name = "HX"  # for use in annotating output data


class ADCRecording(NamedTuple):
    '''
    Times and values of a ADC recording.
    '''
    times: np.ndarray
    values: np.ndarray


class Connection:
    """
    Wrapper for connection to hardware supporting both using a connection
    supplied by the user and constructing an executor from the environment.
    """
    def __init__(self, connection_from_outside: Optional[
            grenade.execution.JITGraphExecutorHandle] = None):
        """
        Construct connection.

        :param connection_from_outside: Optionally supplied connection
        """
        self._conn = connection_from_outside
        self._conn_manager = None

    def get(self):
        """
        Get connection.
        """
        if self._conn is None:
            self._conn_manager = grenade.execution.ManagedJITGraphExecutor()
            self._conn = self._conn_manager.__enter__()  # pylint: disable=unnecessary-dunder-call
        return self._conn

    def end(self):
        """
        Descruct connection if created during instance construction and not
        supplied from the outside.
        """
        if self._conn_manager is not None:
            self._conn_manager.__exit__()


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        int.__init__(n)
        IDMixin.__init__(self)


class GrenadeExperiment(grenade.network.abstract.frontend.Experiment):
    def generate_runtimes(self, runtime: float) -> Dict[
            grenade_common.TimeDomainOnTopology,
            grenade_common.TimeDomainRuntimes]:
        """
        Generate grenade runtimes.

        PyNN only manages one time domain and one batch entry.

        :param runtime: Runtime in ms wall-clock time
        """
        runtime_in_clocks = int(
            runtime * int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        return {
            grenade_common.TimeDomainOnTopology():
            grenade.network.abstract.ClockCycleTimeDomainRuntimes(
                values=[grenade.common.Time(runtime_in_clocks)],
                inter_batch_entry_wait=grenade.common.Time(0))}


class State(BaseState):
    """Represent the simulator state."""

    # pylint: disable=invalid-name
    # TODO: replace by calculation (cf. feature #3594)
    dt: Final[float] = 3.4e-05  # average time between two MADC samples
    # batch entry in grenade data structures, PyNN only uses one batch entry
    batch_entry: Final[int] = 0

    # pylint: disable=invalid-name,too-many-statements
    def __init__(self):
        super().__init__()

        # BSS hardware can only record IrregularlySampledSignal
        self.record_sample_times = True

        self.mpi_rank = 0        # disabled
        self.num_processes = 1   # number of MPI processes
        self.running = False
        self.t = 0
        self.runtimes = []
        self.t_start = 0
        self.min_delay = 0
        self.max_delay = 0
        self.populations: List[Population] = []
        self.projections: List[Projection] = []
        self.recorders = set([])
        self.plasticity_rules: List["PlasticityRule"] = []
        self.recordings = [Recording()]
        self.grenade_experiment = GrenadeExperiment()
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.log = logger.get("pyNN.brainscales2")
        self.initial_config = None
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
        self.ppu_symbols_read = {}
        self.connection = None
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_inside_realtime_begin = None
        self.injection_inside_realtime = None
        self.injection_inside_realtime_end = None
        self.injection_post_realtime = None
        self.execution_instance_data = None
        self.realtime_snippet_count = 0

    def run_until(self, tstop):
        self.add(tstop - self.t)
        self.run()

    def clear(self):
        self.recorders = set([])
        self.populations = []
        self.projections = []
        self.plasticity_rules = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.initial_config = None
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
        self.ppu_symbols_read = {}
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_inside_realtime_begin = None
        self.injection_inside_realtime = None
        self.injection_inside_realtime_end = None
        self.injection_post_realtime = None
        self.execution_instance_data = None

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.runtimes = []
        self.t_start = 0
        self.segment_counter += 1
        self.grenade_experiment.reset()
        self.recordings = [self.recordings[-1]]
        self.realtime_snippet_count = 0

    def _generate_hooks(self):
        assert self.injection_pre_static_config is not None
        assert self.injection_pre_realtime is not None
        assert self.injection_inside_realtime_begin is not None
        assert self.injection_inside_realtime is not None
        assert self.injection_inside_realtime_end is not None
        assert self.injection_post_realtime is not None
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime = sta.AbsoluteTimePlaybackProgramBuilder()
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
        inside_realtime.copy(
            self.injection_inside_realtime)
        inside_realtime_end.copy_back(
            self.injection_inside_realtime_end)
        self._prepare_inside_realtime_end_read(inside_realtime_end)
        post_realtime.copy_back(
            self.injection_post_realtime)
        self._prepare_post_realtime_read(post_realtime)
        return grenade.execution.ExecutionInstanceHooks(
            grenade.common.ChipOnConnection(),
            grenade.execution.ExecutionInstanceHooks.Chip(
                pre_static_config, pre_realtime,
                inside_realtime_begin, inside_realtime,
                inside_realtime_end, post_realtime,
                self.injected_config.ppu_symbols,
                self.injected_readout.ppu_symbols))

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
        self.injection_inside_realtime = self.injected_config.inside_realtime
        self.injection_inside_realtime_end = inside_realtime_end
        self.injection_post_realtime = post_realtime

    def preprocess(self, snippet_begin_time, snippet_end_time):
        """
        Execute all steps needed for the hardware back-end.
        Includes place&route of network graph or execution of calibration.
        Can be called manually to obtain calibration results for e.g.
        CalibHXNeuronCuba/Coba and make adjustments if needed.
        If not called manually is automatically called on run().
        """
        self.grenade_experiment.fill_snippet(
            snippet_begin_time, snippet_end_time, self.connection.get())

    def add(self, runtime: float):
        """
        Adds currently specified configuration, network_graph and inputs to
        list, which is processed serially in run()
        """
        time_begin = time.time()

        snippet_begin_time = deepcopy(self.t)
        self.t += runtime
        self.runtimes.append(runtime)

        self.realtime_snippet_count += 1

        self.grenade_experiment.add_snippet(
            snippet_begin_time, self.t, self.connection.get())

        self.recordings.append(deepcopy(self.recordings[-1]))

        time_after_add = time.time()
        self.log.DEBUG(f"add(): Added {self.realtime_snippet_count}"
                       ". program snippet in "
                       f"{(time_after_add - time_begin):.3f}s")

    def run(self):
        """
        Performs a hardware run of the currently scheduled experiment.
        """
        time_begin = time.time()

        if self.running:
            raise RuntimeError(
                "Call `pynn.reset()` before calling `pynn.run()` again. "
                "BrainScales-2 emulates the behavior of neurons and synapses "
                "in continuous time. Stacking several `pynn.run` commands "
                "without calling `pynn.reset()` between runs is therefore not "
                "supported.")
        self.running = True

        self.grenade_experiment.hooks = {
            grenade.common.ExecutionInstanceOnExecutor():
            self._generate_hooks()}
        self.grenade_experiment.run(self.connection.get())

        self.pre_realtime_read = self._get_pre_realtime_read()
        self.post_realtime_read = self._get_post_realtime_read()
        # The last snippet is the next to be added snippet
        last_executed_snippet_index = -2
        execution_instances = self.grenade_experiment.snippets[
            last_executed_snippet_index].output_data\
            .execution_instances
        if execution_instances\
                .contains(grenade.common.ExecutionInstanceOnExecutor()) \
                and execution_instances.get(
                    grenade.common.ExecutionInstanceOnExecutor()) \
                .read_ppu_symbols \
                and execution_instances.get(
                    grenade.common.ExecutionInstanceOnExecutor())\
                .read_ppu_symbols[self.batch_entry]:
            self.ppu_symbols_read = execution_instances\
                .get(grenade.common.ExecutionInstanceOnExecutor())\
                .read_ppu_symbols[self.batch_entry]
        else:
            self.ppu_symbols_read = {}

        self.execution_instance_data = execution_instances

        self.log.DEBUG("run(): Execution finished in "
                       f"{(time.time() - time_begin):.3f}s")


# state is instantiated in setup()
state: Optional[State] = None
