"""
Implementation of the PyNN API for the neuromorphic BrainScales2 system.
"""
from dataclasses import dataclass, field
import time
from warnings import warn
from enum import Enum, auto
from typing import Dict, Union, Set, Optional
from pyNN import common, space, errors
from pyNN.recording import get_io
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_MIN_DELAY
from pynn_brainscales.brainscales2.connectors import AllToAllConnector, \
    OneToOneConnector, FixedProbabilityConnector, \
    DistanceDependentProbabilityConnector, \
    DisplacementDependentProbabilityConnector, \
    IndexBasedProbabilityConnector, FromListConnector, FromFileConnector, \
    FixedNumberPreConnector, FixedNumberPostConnector, SmallWorldConnector, \
    CSAConnector, CloneConnector, ArrayConnector, FixedTotalNumberConnector
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.standardmodels import cells, synapses
from pynn_brainscales.brainscales2.populations import Population, \
    PopulationView, Assembly
from pynn_brainscales.brainscales2.projections import Projection
from pynn_brainscales.brainscales2.plasticity_rules import Timer, \
    PlasticityRule
from pynn_brainscales.brainscales2 import helper
from dlens_vx_v3 import lola, hal, halco, sta
import pygrenade_vx as grenade
import pylogging as logger
from calix.spiking import SpikingCalibTarget, SpikingCalibOptions


__all__ = ["list_standard_models", "setup", "end", "run", "run_until",
           "run_for", "reset", "initialize", "get_current_time", "create",
           "connect", "set", "record", "logger", "preprocess"]


def list_standard_models():
    """
    Return a list of all the StandardCellType classes available for this
    simulator.
    """
    return [cells.HXNeuron, cells.CalibHXNeuronCuba, cells.CalibHXNeuronCoba]


@dataclass
class InjectedConfiguration():
    """User defined injected configuration

    :param pre_non_realtime: Injection written prior to
                              the non realtime configuration.
    :param pre_realtime: Injection written prior to
                          the realtime configuration.
    :param inside_realtime_begin: Injection written prior to
                          the realtime events.
    :param inside_realtime_end: Injection written after
                          the realtime events and runtime.
    :param post_realtime: Injection written after the
                           the realtime configuration.
    :param ppu_symbols: PPU symbol written during static configuration.
    """
    # TODO: replace hal.Container with union over hal and lola containers
    pre_non_realtime: Union[Dict[halco.Coordinate,
                                 hal.Container], sta.PlaybackProgramBuilder] \
        = field(default_factory=dict)
    pre_realtime: Union[Dict[halco.Coordinate,
                             hal.Container],
                        sta.PlaybackProgramBuilder] = \
        field(default_factory=dict)
    inside_realtime_begin: Union[Dict[halco.Coordinate,
                                      hal.Container],
                                 sta.PlaybackProgramBuilder] = \
        field(default_factory=dict)
    inside_realtime: sta.AbsoluteTimePlaybackProgramBuilder = \
        field(default_factory=sta.AbsoluteTimePlaybackProgramBuilder)
    inside_realtime_end: Union[Dict[halco.Coordinate,
                                    hal.Container],
                               sta.PlaybackProgramBuilder] = \
        field(default_factory=dict)
    post_realtime: Union[Dict[halco.Coordinate,
                              hal.Container], sta.PlaybackProgramBuilder] = \
        field(default_factory=dict)
    ppu_symbols: Dict[str, Union[Dict[halco.HemisphereOnDLS,
                                      hal.PPUMemoryBlock],
                                 lola.ExternalPPUMemoryBlock]] = \
        field(default_factory=dict)


@dataclass
class InjectedReadout():
    """User defined injected readout

    :param pre_realtime: Injection of reads after the
                           the pre_realtime configuration.
    :param inside_realtime_begin: Injection of reads after the
                           the inside_realtime_begin configuration.
    :param inside_realtime_end: Injection of reads after the
                           the inside_realtime_end configuration.
    :param post_realtime: Injection of reads after the
                           the post_realtime configuration.
    :param ppu_symbols: PPU symbol read after inside_realtime_end.
    """
    pre_realtime: Set[halco.Coordinate] = field(default_factory=set)
    inside_realtime_begin: Set[halco.Coordinate] = field(default_factory=set)
    inside_realtime_end: Set[halco.Coordinate] = field(default_factory=set)
    post_realtime: Set[halco.Coordinate] = field(default_factory=set)
    ppu_symbols: Set[str] = field(default_factory=set)


class RunCommand(Enum):
    """Command specifying the type of run to be performed in pynn.run()

    :param prepare: Command that only induces all preparatory steps for the
                        hardware backend, e.g. place&route of network graph or
                        execution of calibration.
    :param append: Command that induces the scheduling of an experiment snippet
                        for the system in the currently defined state for a
                        certain runtime.
    :param execute: Command that induces a hardware run. Executes the already
                        scheduled experiment and optionally append one last
                        experiment snippet.
    """
    PREPARE = auto()
    APPEND = auto()
    EXECUTE = auto()


# TODO: handle the delays (cf. feature #3657)
def setup(timestep=simulator.State.dt, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):
    """
    Should be called at the very beginning of a script.
    :param extra_params:
        most params come from pynn.common.setup
        neuronPermutation: List providing lookup for custom pyNN neuron to
                           hardware neuron. Index: HW related population neuron
                           enumeration. Value: HW neuron enumeration. Can be
                           shorter than total HW neuron count. E.g. [2,4,5]
                           results in the first neuron of the first HXNeuron
                           population to be assigned to
                           AtomicNeuronOnDLS(Enum(2)) and so forth.
        backgroundPermutation: List providing lookup for custom pyNN background
                               spike source to hardware entity.
                               Index: HW related population source
                               enumeration. Value: HW source enumeration. Can
                               be shorter than total HW source count. E.g.
                               [2,3] results in the first population to be
                               assigned to PADIBusOnPADIBusBlock(2) and so
                               forth.
        enable_neuron_bypass: Enable neuron bypass mode: neurons forward spikes
                              arriving at the synaptic input (i.e. no leaky
                              integration is happening); defaults to False.
        initial_config: Initial configuration of the entire chip. Can for
                        example be used to manually apply a calibration
                        result.
        injected_config: Optional user defined injected configuration.
        injected_readout: Optional user defined injected readout.
        calibration_cache: Directory where automated calibration is cached.
                           If none provided defaults to home cache.
        injected_calib_target: Optional user defined injected calibration
                               target, which may be partially overwritten by
                               parameters set at populations and projections.
        injected_calib_options: Optional user defined injected calibration
                                options, which may be partially overwritten by
                                parameters set at populations and projections.
    """
    time_begin = time.time()

    # global instance singleton
    simulator.state = simulator.State()

    max_delay = extra_params.pop('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    if min_delay == "auto":
        min_delay = 0
    if max_delay == "auto":
        max_delay = 0
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.neuron_placement = simulator.NeuronPlacement(
        extra_params.pop("neuronPermutation",
                         simulator.NeuronPlacement.DEFAULT_PERMUTATION))
    simulator.state.background_spike_source_placement = \
        simulator.BackgroundSpikeSourcePlacement(
            extra_params.pop("backgroundPermutation",
                             simulator.BackgroundSpikeSourcePlacement
                             .DEFAULT_PERMUTATION))
    simulator.state.injected_config = \
        extra_params.pop('injected_config', InjectedConfiguration())
    simulator.state.injected_readout = \
        extra_params.pop('injected_readout', InjectedReadout())
    simulator.state.conn = extra_params.pop('connection', None)
    simulator.state.conn_comes_from_outside = simulator.state.conn is not None
    initial_config = extra_params.pop('initial_config', None)
    enable_neuron_bypass = extra_params.pop('enable_neuron_bypass', False)
    if enable_neuron_bypass:
        if initial_config is not None:
            simulator.state.log.INFO(
                "setup(): Supplied initial_config "
                "overwritten by enable_neuron_bypass")
        initial_config = lola.Chip.default_neuron_bypass
    simulator.state.initial_config = initial_config
    simulator.state.prepare_static_config()
    simulator.state.calib_cache_dir = extra_params.pop('calibration_cache',
                                                       None)
    simulator.state.injected_calib_target = \
        extra_params.pop('injected_calib_target', SpikingCalibTarget())
    simulator.state.injected_calib_options = \
        extra_params.pop('injected_calib_options', SpikingCalibOptions())

    if extra_params:
        raise KeyError("unhandled extra_params in call to pynn.setup(...):"
                       f"{extra_params}")

    simulator.state.log.DEBUG(
        f"setup(): Completed in {(time.time() - time_begin):.3f}s")


def end():
    """Do any necessary cleaning up before exiting."""
    time_begin = time.time()
    for (population, variables, filename) in simulator.state.write_on_end:
        io_file = get_io(filename)
        population.write_data(io_file, variables)
    simulator.state.write_on_end = []

    if not simulator.state.conn_comes_from_outside and \
       simulator.state.conn_manager is not None:
        simulator.state.conn_manager.__exit__()
        simulator.state.conn_manager = None
        assert simulator.state.conn is not None
        simulator.state.conn = None
    simulator.state.log.DEBUG(
        f"end(): Completed in {(time.time() - time_begin):.3f}s")
    # remove instance singleton
    simulator.state = None


# common.build_run's run (first return value) performs some arithmetic (plus)
# on run's runtime parameter and finally dispatches to run_until; to support
# runtime=`None` we need to directly dispatch to our own run function.
_, run_until = common.build_run(simulator)


def run(runtime: Optional[float],
        command: RunCommand = RunCommand.EXECUTE):
    """
    Take different actions according to the value passed for command:

    For command=pyNN.RunCommand.PREPARE:
    Execute all steps needed for the hardware back-end.
    Includes place&route of network graph or execution of calibration.
    Can be called manually to obtain calibration results for e.g.
    CalibHXNeuron and make adjustments if needed.
    If not called manually is automatically called on
    command=pyNN.RunCommand.APPEND.

    For command=pyNN.RunCommand.APPEND:
    Make a snapshot of the current network configuration and cache it
    together with the according runtime, for which the system shall evolve
    in this configuration.
    Append snippet (=certain network description valid for certain
    duration) to currently scheduled experiment.

    For command=pyNN.RunCommand.EXECUTE:
    Execute the experiment that was already scheduled by using
    command=pyNN.RunCommand.APPEND.
    Can also append a last snippet to the experiment right before
    executing the experiment, if a value is passed for runtime.
    """
    if command == RunCommand.PREPARE:
        simulator.state.preprocess(0, 0)
    elif command == RunCommand.APPEND:
        if runtime is None:
            raise ValueError("Appending a configuration that is executed for "
                             "no time is forbidden.")
        simulator.state.add(runtime)
    elif command == RunCommand.EXECUTE:
        if runtime is not None:
            simulator.state.add(runtime)
        elif not simulator.state.runtimes:
            raise ValueError(
                "No experiment scheduled for execution yet.\n"
                "To schedule an execution of the currently defined network "
                "configuration use `command=pyNN.RunCommand.APPEND`.\n"
                "To only perform map and route operations on the current "
                "network configuration use `command=pyNN.RunCommand.PREPARE`")
        simulator.state.run()
    else:
        raise ValueError("Passed command is invalid.")


run_for = run


def preprocess():
    """
    Does the same as pyNN.run(pyNN.RunCommand.PREPARE).
    Exists only due to backwards-compatibility reasons.
    """
    warn('Please use pyNN.run(None, pyNN.RunCommand.PREPARE)',
         DeprecationWarning, stacklevel=2)
    simulator.state.preprocess(0, 0)


reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
    num_processes, rank = common.build_state_queries(simulator)

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector,
                               synapses.StaticSynapse)

# pylint: disable=redefined-builtin
set = common.set

record = common.build_record(simulator)


def get_post_realtime_read() -> Dict[halco.Coordinate, hal.Container]:
    """
    Get injected read results of after post_realtime section.
    :return: Dictionary with coordinates as keys and read container as
             values.
    """
    if not simulator.state:
        raise RuntimeError("Post-realtime reads are only available with valid"
                           " simulator after calling setup().")
    return simulator.state.post_realtime_read


def get_pre_realtime_read() -> Dict[halco.Coordinate, hal.Container]:
    """
    Get injected read results of after pre_realtime section.
    :return: Dictionary with coordinates as keys and read container as
             values.
    """
    if not simulator.state:
        raise RuntimeError("Pre-realtime reads are only available with valid"
                           " simulator after calling setup().")
    return simulator.state.pre_realtime_read


# pylint: disable=invalid-name
def get_post_realtime_read_ppu_symbols() -> Dict[
        str, Union[Dict[halco.HemisphereOnDLS,
                        hal.PPUMemoryBlock],
                   lola.ExternalPPUMemoryBlock]]:
    """
    Get injected PPU symbol read results of after inside_realtime_end section.
    :return: Dictionary with symbol name as keys and read container(s) as
             values.
    """
    if not simulator.state:
        raise RuntimeError(
            "Post-realtime PPU symbol reads are only available with valid "
            "simulator after calling setup().")
    if not simulator.state.running:
        raise RuntimeError(
            "Read PPU symbols are only available after emulation on hardware. "
            "Call pynn.run(...).")
    return simulator.state.ppu_symbols_read


def get_backend_statistics() \
        -> grenade.network.NetworkGraphStatistics:
    """
    Get statistics of placement and routing like amount of time spent and
    number of hardware entities used.
    :raises RuntimeError: If the simulator is not active, i.e. pynn.setup()
                          was not called.
    :raises RuntimeError: If the routing and placement step were not
                          performed, i.e. pynn.run() was not called.
    :return: Statistics object.
    """
    if not simulator.state:
        raise RuntimeError(
            "Backend statistics are only available for active "
            "simulator after calling setup().")
    if not simulator.state.grenade_network_graph:
        raise RuntimeError(
            "Backend statistics are only available after first mapping and"
            " routing execution, which happens in pynn.run().")
    return grenade.network.extract_statistics(
        simulator.state.grenade_network_graph)


def get_execution_time_info() -> grenade.signal_flow.ExecutionTimeInfo:
    """
    Get time information of last execution.
    :raises RuntimeError: If the simulator is not active, i.e. pynn.setup()
                          was not called.
    :raises RuntimeError: If no info is available, i.e. pynn.run() was not
                          called.
    :return: Time info object.
    """
    if not simulator.state:
        raise RuntimeError(
            "Execution time information is only available for active "
            "simulator after calling setup()."
        )
    if simulator.state.execution_time_info is None:
        raise RuntimeError(
            "Execution time information is only available after the hardware "
            "configuration has been determined. Call pynn.run(...) (you can "
            "pass `None` as an argument if want to determine the hardware "
            "configuration without an emulation on hardware.")
    return simulator.state.execution_time_info


def get_execution_health_info() -> grenade.signal_flow.ExecutionHealthInfo:
    """
    Get health information of last execution.

    :raises RuntimeError: If the simulator is not active, i.e. pynn.setup()
                          was not called.
    :raises RuntimeError: If no info is available, i.e. pynn.run() was not
                          called.
    :return: Health info object.
    """
    if not simulator.state:
        raise RuntimeError(
            "Execution health information is only available for active "
            "simulator after calling setup()."
        )
    if simulator.state.execution_time_info is None:
        raise RuntimeError(
            "Execution health information is only available after the "
            "hardware configuration has been determined. Call pynn.run(...) "
            "(you can pass `None` as an argument if want to determine the "
            "hardware configuration without an emulation on hardware.")
    return simulator.state.execution_health_info
