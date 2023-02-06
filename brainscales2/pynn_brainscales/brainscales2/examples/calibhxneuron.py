#!/usr/bin/env python

import pynn_brainscales.brainscales2 as pynn


def main():
    pynn.setup()

    pop = pynn.Population(4, pynn.cells.CalibHXNeuronCuba(
        v_rest=[80, 50, 70, 60],
        v_reset=71,
        v_thresh=125
    ))

    pop.record(["spikes"])

    exc_stim_pop = pynn.Population(
        10, pynn.standardmodels.cells.SpikeSourcePoisson(
            rate=1e4, start=0, duration=10))

    pynn.Projection(
        exc_stim_pop, pop, pynn.AllToAllConnector(),
        synapse_type=pynn.standardmodels.synapses.StaticSynapse(weight=63),
        receptor_type="excitatory")

    # new pre-process call, applies map&route and calibration
    # is needed to work on the result of the calibration, is also implicitly
    # called upon run()
    pynn.preprocess()

    # get calibration result in form of list of atomic neurons indexed as the
    # population
    # we store the calibration targets and results as convenience for the user
    print(pop.calib_target)
    print(pop.calib_hwparams)

    # modify parameters, e.g. leak other threshold
    pop.actual_hwparams[0].leak.v_leak = 1022

    pop[1:3].actual_hwparams[1].leak.v_leak = 1022

    pynn.run(10)

    spikes = pop.get_data("spikes").segments[0].spiketrains
    return spikes


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    hxcommlogger = pynn.logger.get("hxcomm")
    pynn.logger.set_loglevel(hxcommlogger, pynn.logger.LogLevel.WARN)
    log = pynn.logger.get("CalibHXNeuronCuba")

    spiketrains = main()

    log.INFO("Expect higher rates for neurons 0 and 2")
    for index, train in enumerate(spiketrains):
        log.INFO(f"Number of spikes neuron {index}: {len(train)}")
