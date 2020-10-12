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
from dlens_vx_v2 import halco
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


# TODO: handle the delays (cf. feature #3657)
def setup(timestep=simulator.state.dt, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):
    """
    Should be called at the very beginning of a script.
    :param extra_params:
        most params come from pynn.common.setup
        neuronPermutation: List providing lookup for custom pyNN neuron to
                           hardware neuron. Can be shorter than total HW neuron
                           count.
        enable_neuron_bypass: Enable neuron bypass mode: neurons forward spikes
                              arriving at the synaptic input (i.e. no leaky
                              integration is happening); defaults to False.
    """

    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    simulator.state.neuron_placement = simulator.\
        state.check_neuron_placement_lut(
            extra_params.get("neuronPermutation", range(
                halco.AtomicNeuronOnDLS.size))
        )
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


def end():
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io_file = get_io(filename)
        population.write_data(io_file, variables)
    simulator.state.write_on_end = []


run, run_until = common.build_run(simulator)
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
