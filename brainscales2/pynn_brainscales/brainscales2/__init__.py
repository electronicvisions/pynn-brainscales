from dataclasses import dataclass, field
from typing import Dict, Union
from pyNN import common, space
from pyNN.recording import get_io
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_MIN_DELAY
from pyNN.connectors import AllToAllConnector, OneToOneConnector, \
    FixedProbabilityConnector, DistanceDependentProbabilityConnector, \
    DisplacementDependentProbabilityConnector, \
    IndexBasedProbabilityConnector, FromListConnector, FromFileConnector, \
    FixedNumberPreConnector, FixedNumberPostConnector, SmallWorldConnector, \
    CSAConnector, CloneConnector, ArrayConnector, FixedTotalNumberConnector
from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.standardmodels import cells, synapses
from pynn_brainscales.brainscales2.populations import Population, \
    PopulationView, Assembly
from pynn_brainscales.brainscales2.projections import Projection
from pynn_brainscales.brainscales2 import helper
from dlens_vx_v2 import hal, halco
import pylogging as logger


__all__ = ["list_standard_models", "setup", "end", "run", "run_until",
           "run_for", "reset", "initialize", "get_current_time", "create",
           "connect", "set", "record", "logger"]


def list_standard_models():
    """
    Return a list of all the StandardCellType classes available for this
    simulator.
    """
    return [cells.HXNeuron]


@dataclass
class InjectedConfiguration():
    """User defined injected configuration

    :param pre_non_realtime: A dictionary of key=coordinate and
                              value=containers entries written prior to
                              the non realtime configuration.
    :param post_non_realtime: A dictionary of key=coordinate and
                               value=containers entries written after the
                               the non realtime configuration.
    :param pre_realtime: A dictionary of key=coordinate and
                          value=containers entries written prior to
                          the realtime configuration.
    :param post_realtime: A dictionary of key=coordinate and
                           value=containers entries written after the
                           the realtime configuration.
    """
    # TODO: replace hal.Container with union over hal and lola containers
    pre_non_realtime: Dict[halco.Coordinate,
                           hal.Container] = field(default_factory=dict)
    post_non_realtime: Dict[halco.Coordinate,
                            hal.Container] = field(default_factory=dict)
    pre_realtime: Dict[halco.Coordinate,
                       hal.Container] = field(default_factory=dict)
    post_realtime: Dict[halco.Coordinate,
                        hal.Container] = field(default_factory=dict)


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
        enable_neuron_bypass: Enable neuron bypass mode: neurons forward spikes
                              arriving at the synaptic input (i.e. no leaky
                              integration is happening); defaults to False.
        injected_config: Optional user defined injected configuration.
    """

    # global instance singleton
    simulator.state = simulator.State()

    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    enable_neuron_bypass = extra_params.get('enable_neuron_bypass', False)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    if min_delay == "auto":
        min_delay = 0
    if max_delay == "auto":
        max_delay = 0
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.enable_neuron_bypass = enable_neuron_bypass
    simulator.state.neuron_placement = simulator.NeuronPlacement(
        extra_params.get("neuronPermutation",
                         simulator.NeuronPlacement.default_permutation))
    simulator.state.injected_config = \
        extra_params.get('injected_config', InjectedConfiguration())


def end():
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io_file = get_io(filename)
        population.write_data(io_file, variables)
    simulator.state.write_on_end = []

    # remove instance singleton
    simulator.state = None


# common.build_run's run (first return value) performs some arithmetic (plus)
# on run's runtime parameter and finally dispatches to run_until; to support
# runtime=`None` we need to directly dispatch to our own run function.
_, run_until = common.build_run(simulator)


def run(*args, **kwargs):
    return simulator.state.run(*args, **kwargs)


run_for = run

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
