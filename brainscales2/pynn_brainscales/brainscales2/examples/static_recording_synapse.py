#!/usr/bin/env python

import numpy as np
import pynn_brainscales.brainscales2 as pynn


def main():
    pynn.setup()

    nrn = pynn.Population(1, pynn.cells.HXNeuron())
    nrn.record(["spikes", "v"])

    spike_times = np.arange(0, 10, 0.01)
    pop_input = pynn.Population(1, pynn.cells.SpikeSourceArray,
                                cellparams={"spike_times": spike_times})

    timer = pynn.Timer(start=5, period=10, num_periods=1)
    synapse = pynn.standardmodels.synapses.StaticRecordingSynapse(
        timer=timer, weight=63, observables={"weights"})

    projection = pynn.Projection(
        pop_input, nrn, pynn.AllToAllConnector(),
        synapse_type=synapse)

    pynn.run(10)

    weights = projection.get_data("weights")

    pynn.end()

    return weights


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    log = pynn.logger.get("static_recording_synapse")
    data = main()

    log.INFO("Recorded weights of projection: ", data)
