from typing import Optional
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2.utils.chip_plotter import ChipPlotter
import pynn_brainscales.brainscales2 as pynn


# pylint: disable = R0914
def setup_synfire_chain(
        alternating: bool,
        exc_pop_sizes: int,
        inh_pop_sizes: int,
        numb_pops: int,
        closed: Optional[bool] = False):
    '''
    sets up a synfire chain network in pynn
    '''
    pynn.setup()
    # synfire chain
    # parameters to play around with
    pop_sizes = {"exc": exc_pop_sizes, "inh": inh_pop_sizes}
    pop_collector = {"exc": [], "inh": []}
    neuron_params = {"refractory_period_refractory_time": 50}
    pop_type_to_id = {"exc": [], "inh": []}
    # stuff
    count = 0
    if alternating:
        for pop_num in range(numb_pops):
            for syn_type in ["exc", "inh"]:
                pop = pynn.Population(
                    pop_sizes[syn_type],
                    pynn.cells.HXNeuron(**neuron_params),
                    label=syn_type + " pop " + str(pop_num),
                )
                pop.record(["spikes"])
                pop_collector[syn_type].append(pop)
                pop_type_to_id[syn_type].append(count)
                count += 1
    else:
        for syn_type in ["exc", "inh"]:
            for pop_num in range(numb_pops):
                pop = pynn.Population(
                    pop_sizes[syn_type],
                    pynn.cells.HXNeuron(**neuron_params),
                    label=syn_type + " pop " + str(pop_num),
                )
                pop.record(["spikes"])
                pop_collector[syn_type].append(pop)
                pop_type_to_id[syn_type].append(count)
                count += 1

    stim_pop = pynn.Population(
        pop_sizes["exc"],
        pynn.cells.SpikeSourceArray(spike_times=[0]),
        label="stim pop",
    )
    population_contrast_groups = {}
    population_contrast_groups = pop_type_to_id
    population_contrast_groups["stim"] = [count]

    proj_collector = {"exc_exc": [], "exc_inh": [], "inh_exc": []}

    # connect stim -> exc
    proj_collector["stim_exc"] = pynn.Projection(
        stim_pop,
        pop_collector["exc"][0],
        pynn.AllToAllConnector(),
        synapse_type=StaticSynapse(weight=0),
        receptor_type="excitatory",
    )

    # connect stim -> inh
    proj_collector["stim_inh"] = pynn.Projection(
        stim_pop,
        pop_collector["inh"][0],
        pynn.AllToAllConnector(),
        synapse_type=StaticSynapse(weight=0),
        receptor_type="excitatory",
    )

    for pop_index in range(numb_pops):
        # connect inh -> exc
        proj_collector["inh_exc"].append(
            pynn.Projection(
                pop_collector["inh"][pop_index],
                pop_collector["exc"][pop_index],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=0),
                receptor_type="inhibitory",
            )
        )

        # if synfire chain is not closed, the last exc -> exc and exc -> inh
        # that connects back to the first population needs to be skipped.
        if (pop_index == numb_pops - 1) and not closed:
            continue

        # connect exc -> exc
        proj_collector["exc_exc"].append(
            pynn.Projection(
                pop_collector["exc"][pop_index],
                pop_collector["exc"][(pop_index + 1) % numb_pops],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=0),
                receptor_type="excitatory",
            )
        )

        # connect exc -> inh
        proj_collector["exc_inh"].append(
            pynn.Projection(
                pop_collector["exc"][pop_index],
                pop_collector["inh"][(pop_index + 1) % numb_pops],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=0),
                receptor_type="excitatory",
            )
        )

    pynn.simulator.state.preprocess(0, 0)
    return population_contrast_groups


if __name__ == "__main__":
    # setup pynn network
    pop_contrast = setup_synfire_chain(True, 16, 16, 5)
    # create chipplotter object, define default params
    chipPlotter = ChipPlotter(population_contrast_groups=pop_contrast)
    # define which kinds of plots you'd like
    # you can alter params from default params for each individual plot
    chipPlotter.add_plot_chip_map(
        separate_view=False,
        sort_synapse_by_padi_bus=True)
    chipPlotter.add_plot_chip_map(
        separate_view=True,
        sort_synapse_by_padi_bus=True)
    chipPlotter.add_plot_matrix(plot_each_neuron=False)
    chipPlotter.add_plot_matrix(plot_each_neuron=True)
    # do the plotting
    chipPlotter.plot_all()
    # save the plots
    chipPlotter.save_all(folder="chip_plotter_usage_example/", formats=["png"])
