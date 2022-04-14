#!/usr/bin/env python

from typing import Dict
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v2 import halco, hal


def main(num_spikes: int = 200, runtime: float = 20.) \
        -> Dict[halco.CrossbarOutputOnDLS, int]:
    """
    This example shows readout of the event output counters in the routing
    crossbar via injected reads.
    :param num_spikes: Number of spikes to inject during the experiment.
    :param runtime: Runtime of the experiment [ms].
    :return: Difference between counter values before the experiment and after
             the experiment. The results are saved in a dictionary with an
             entry for each crossbar output.
    """
    # To readout the event counters, we inject read commands before and
    # after the realtime experiment to compare the counter values.
    injected_readout = pynn.InjectedReadout()
    for coord in halco.iter_all(halco.CrossbarOutputOnDLS):
        injected_readout.pre_realtime.add(coord)
        injected_readout.post_realtime.add(coord)

    # Additionally, we need to enable the event counters via the injected
    # configuration.
    injected_config = pynn.InjectedConfiguration()
    crossbar_output_config = hal.CrossbarOutputConfig()
    for coord in halco.iter_all(halco.CrossbarOutputOnDLS):
        # pylint: disable=unsupported-assignment-operation
        crossbar_output_config.enable_event_counter[coord] = True
    injected_config.pre_realtime = \
        {halco.CrossbarOutputConfigOnDLS(): crossbar_output_config}

    # Injections are specified upon setup of PyNN.
    pynn.setup(injected_config=injected_config,
               injected_readout=injected_readout,
               enable_neuron_bypass=True)

    # The network consists of a single-neuron input and output population,
    nrn = pynn.Population(1, pynn.cells.HXNeuron())
    nrn.record(["spikes"])

    pop_input = pynn.Population(
        1, pynn.cells.SpikeSourceArray,
        cellparams={
            "spike_times": np.linspace(0.01, runtime - 0.01, num_spikes)})

    # connected via one synapse.
    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
    pynn.Projection(pop_input, nrn, pynn.OneToOneConnector(),
                    synapse_type=synapse)

    pynn.run(runtime)

    # After execution, the read results are available.
    event_counter_reads_before = pynn.get_pre_realtime_read()
    event_counter_reads_after = pynn.get_post_realtime_read()

    # We calculate the difference in event counter values and return
    # the result.
    diffs = dict()
    for coord in halco.iter_all(halco.CrossbarOutputOnDLS):
        c_before = event_counter_reads_before[coord]
        c_after = event_counter_reads_after[coord]
        difference = c_after.value - c_before.value
        diffs.update({coord: int(difference)})

    pynn.end()
    return diffs


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    log = pynn.logger.get("crossbar_event_counter_read")
    ds = main()

    for c, diff in ds.items():
        log.INFO("Event counter diff {}: {}".format(c, diff))
