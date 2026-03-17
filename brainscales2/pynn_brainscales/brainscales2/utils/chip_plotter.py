# pylint: disable = C0302
import os
from typing import Optional
from enum import Enum
import colorsys
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v3 import halco


class PlotTypes(Enum):
    """
    plot_type:
        SANDWICH: chipmap showing neuron and synapse placement all on one plot
        SEPARATE: chipmap showing neuron placement in upper subplot
        and synapse placement in lower subplot
        POPULATIONMATRIX: showing how populations are connected
        NEURONMATRIX: showing how individual neurons are connected
    """

    SANDWICH = 1
    SEPARATE = 2
    POPULATIONMATRIX = 3
    NEURONMATRIX = 4


class ChipPlotter:
    """
    How to use:
    call __init__(**params)
    call add_plot_matrix(**params) or add_plot_chipmap(**params)
        to add plots to plot_list
    call plot_all() or plot_by_id(id : int) to plot plots from
        plot_list
    call save_all(**params) or save_by_id(**params) to save plots
        from plot_list
    plot_list syntax: [{"plot_type": plot_type, "fig":fig,
        "ax":ax, "settings":settings},
                        {"plot_type": plot_type, "fig":fig,
        "ax":ax, "settings":settings},
                        ...]
    """

    # pylint: disable = too-many-arguments
    def __init__(
        self,
        do_label_synapses: Optional[bool] = False,
        has_grid: Optional[bool] = True,
        color_is_source: Optional[bool] = True,
        alpha: Optional[float] = 0.9,
        custom_title: Optional[str] = None,
        population_contrast_groups: Optional[dict] = None,
    ):
        """
        Defaultsettings for when adding plots.
        Args:
        :param do_label_synapses: (bool) add labels for each
            projection in plot legend
        :param has_grid: (bool) add grid to plot
        :param color_is_source: (bool) color of synapses corresponds
            to color of presynaptic population
        :param alpha: (float) opacity of data points on plot
        :param custom_title: (string) set custom title for plot
        :param population_contrast_groups: (list<dict<string:list<int>>>)
            separate population ids into different groups that show
            with contrasting colors on the plot
        """

        self.default_settings = {
            "do_label_synapses": do_label_synapses,
            "has_grid": has_grid,
            "color_is_source": color_is_source,
            "alpha": alpha,
            "custom_title": custom_title,
            "population_contrast_groups": population_contrast_groups,
        }
        self.plot_list = []
        self.neuron_matrix = None
        self.synapse_map = None
        self.unused_rows = None
        self.pop_matrix = None
        self.ext_ids = None
        self.external_pops_id = None
        self.neuron_map = None
        self.pops = None
        self.projections = None

    # pylint: disable = too-many-locals, line-too-long
    def add_plot_chip_map(
        self,
        do_label_synapses: Optional[bool] = None,
        has_grid: Optional[bool] = None,
        color_is_source: Optional[bool] = None,
        alpha: Optional[float] = None,
        custom_title: Optional[str] = None,
        population_contrast_groups: Optional[dict] = None,
        neuron_height: Optional[int] = 50,
        center: Optional[tuple] = None,
        sort_synapse_by_padi_bus: Optional[bool] = True,
        highlight_empty_rows: Optional[bool] = True,
        separate_view: Optional[bool] = False,
    ):
        """
        adds chipmap plot to plotlist
        default setting will be used if argument
            not specified
        Args:
        :param do_label_synapses: (bool) add labels for
            each projection in plot legend
        :param has_grid: (bool) add grid to plot
        :param color_is_source: (bool): color of synapses
            corresponds to color of presynaptic population
        :param alpha: (float) opacity of data points on plot
        :param custom_title: (string) set custom title for plot
        :param population_contrast_groups:
            (list<dict<string:list<int>>>) separate population
        :param ids: into different groups that show with
            contrasting colors on the plot
        :param neuron_height: (int) height of neuron rows in units of synapse rows
            on plot if separate view false
        :param center: (list<int,int>) center of plot if separate
            view false
        :param sort_synapse_by_padi_bus: (bool) if false,
            synapse y coordinate corresponds
            to physical location. If true, synapse y
            corresponds to location on padibus
        :param highlight_empty_rows: (bool) highlight unused
            synapse rows in grey
        :param separate_view: (bool) plot neuron and synapse
            placement in 2 subplots or together
        """
        if center is None:
            center = [0, 0]
        if do_label_synapses is None:
            do_label_synapses = self.default_settings["do_label_synapses"]
        if has_grid is None:
            has_grid = self.default_settings["has_grid"]
        if color_is_source is None:
            color_is_source = self.default_settings["color_is_source"]
        if alpha is None:
            alpha = self.default_settings["alpha"]
        if custom_title is None:
            custom_title = self.default_settings["custom_title"]
        if population_contrast_groups is None:
            population_contrast_groups = self.default_settings[
                "population_contrast_groups"
            ]
        plot_type = PlotTypes.SEPARATE if separate_view else PlotTypes.SANDWICH
        settings = {
            "do_label_synapses": do_label_synapses,
            "has_grid": has_grid,
            "color_is_source": color_is_source,
            "alpha": alpha,
            "custom_title": custom_title,
            "population_contrast_groups": population_contrast_groups,
            "neuron_height": neuron_height,
            "center": center,
            "sort_synapse_by_padi_bus": sort_synapse_by_padi_bus,
            "highlight_empty_rows": highlight_empty_rows,
            "plot_type": plot_type,
        }
        if separate_view:
            fig = plt.figure()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
            ax = [ax1, ax2]
        else:
            fig, ax = plt.subplots()
        self.plot_list.append(
            {"plot_type": plot_type, "fig": fig,
             "ax": ax, "settings": settings}
        )

    def retrieve_data(
        self, neuron_map: bool, synapse_map: bool,
        pop_matrix: bool, neuron_matrix: bool
    ) -> dict[str, list]:
        """
        gathers data
        :param neuron_map: (bool) retrieve neuron map, syntax:
            [[pop0 neuron0 pos, pop0 neuron1 pos,
            pop0 neuron...], [pop1 neuron0 pos, pop1 neuron1 pos,
            pop1 neuron...],[pop...]]
            retrieve external pops list, syntax:
            [[popindex, popsize,
            neuroncount until this pop],...]
        :param synapse_map: (bool) retrieve synapse map, syntax:
            [[[pop1,pop2],[synapse_x,synapse_y],
            [synapse_x,synapse_y],...],...]

        :param pop_matrix: (bool) retrieve population matrix,
            syntax: [[proj0,[pre0_index,post0_index]],
            [proj1,[pre1_index,post1_index]], ...]

        :param neuron_matrix: (bool) retrieve neuron matrix,
            syntax:
            [[proj0,[pre0_index,post0_index],[[neuron0,neuron1],
            [neuron1,neuron5]]], [proj1,[pre1_index,post1_index],
            [],[[neuron0,neuron1],[neuron1,neuron5],[neuron2,neuron5]]], ...]
        """
        self.pops = pynn.simulator.state.populations
        self.projections = pynn.simulator.state.projections
        output = {}
        if neuron_map:
            neuron_info = self.__create_neuron_map()
            output["neuron_map"] = neuron_info["neuron_map"]
            output["external_pop_ids"] = neuron_info["external_pop_ids"]
        if synapse_map:
            synapse_info = self.__create_synapse_map()
            output["synapse_map"] = synapse_info["synapse_map"]
            output["unused_rows"] = synapse_info["unused_rows"]
        if pop_matrix:
            output["pre_post_pop_matrix"] = self.__create_pop_matrix()
        if neuron_matrix:
            output["pre_post_pop_matrix"] = self.__create_pop_matrix()
            output["pre_post_neuron_matrix"] = self.__create_neuron_matrix()
        return output

    def add_plot_matrix(
        self,
        do_label_synapses: Optional[bool] = None,
        has_grid: Optional[bool] = None,
        color_is_source: Optional[bool] = None,
        alpha: Optional[float] = None,
        custom_title: Optional[str] = None,
        population_contrast_groups: Optional[dict] = None,
        plot_each_neuron: Optional[bool] = False,
    ):
        """
        adds matrix plot to plotlist
        default setting will be used if argument not specified
        Args:
        :param do_label_synapses: (bool) add labels for each
            projection in plot legend
        :param has_grid: (bool) add grid to plot
        :param color_is_source: (bool) color of synapses corresponds
            to color of presynaptic population
        :param alpha: (float) opacity of data points on plot
        :param custom_title: (string) set custom title for plot
        :param population_contrast_groups: (list<dict<string:list<int>>>)
            separate population ids into different groups that show
            with contrasting colors on the plot
        :param plot_each_neuron: (bool) plot connections between each
            individual neuron, or just populations
        """
        if do_label_synapses is None:
            do_label_synapses = self.default_settings["do_label_synapses"]
        if has_grid is None:
            has_grid = self.default_settings["has_grid"]
        if color_is_source is None:
            color_is_source = self.default_settings["color_is_source"]
        if alpha is None:
            alpha = self.default_settings["alpha"]
        if custom_title is None:
            custom_title = self.default_settings["custom_title"]
        if population_contrast_groups is None:
            population_contrast_groups = self.default_settings[
                "population_contrast_groups"
            ]
        plot_type = (
            PlotTypes.NEURONMATRIX if plot_each_neuron else PlotTypes.POPULATIONMATRIX
        )
        settings = {
            "do_label_synapses": do_label_synapses,
            "has_grid": has_grid,
            "color_is_source": color_is_source,
            "alpha": alpha,
            "custom_title": custom_title,
            "population_contrast_groups": population_contrast_groups,
            "plot_type": plot_type,
        }
        fig, ax = plt.subplots()
        self.plot_list.append(
            {"plot_type": plot_type, "fig": fig, "ax": ax, "settings": settings}
        )

    def __create_neuron_map(self):
        """neuron map syntax: [[pop0 neuron0 pos, pop0 neuron1 pos,
        pop0 neuron...], [pop1 neuron0 pos, pop1 neuron1 pos,
        pop1 neuron...], [pop...]]
        external pops list syntax [[popindex,
            popsize, neuroncount until this pop],...]
        """
        pops_allcells = []
        for pop in self.pops:
            pops_allcells.append(pop.all_cells)
        external_pops = []
        pops_pos = []
        for i, _ in enumerate(self.pops):
            if not isinstance((self.pops[i]).celltype, pynn.cells.ExternalNeuron):
                pop = pops_allcells[i]
                logical_neurons = (
                    pynn.simulator.state.neuron_placement.id2logicalneuron(pop)
                )
                pop_pos = []
                for neuron in logical_neurons:
                    x_coord = int(neuron.get_atomic_neurons()[0].toNeuronColumnOnDLS())
                    y_coord = int(neuron.get_atomic_neurons()[0].toNeuronRowOnDLS())
                    pop_pos.append([x_coord, y_coord])
                pops_pos.append(pop_pos)
            else:
                pops_pos.append([])
                size_until_here = 0
                if len(external_pops) > 0:
                    size_until_here = (
                        external_pops[len(external_pops) - 1][2]
                        + external_pops[len(external_pops) - 1][1]
                    )
                external_pops.append([i, self.pops[i].size, size_until_here])
        self.neuron_map = pops_pos
        self.external_pops_id = external_pops
        self.ext_ids = [element[0] for element in self.external_pops_id]
        return {"neuron_map": pop_pos, "external_pop_ids": external_pops}

    def __create_synapse_map(self):
        """synapse map syntax: [[[pop1,pop2],
        [synapse_x,synapse_y],[synapse_x,synapse_y],...]
        ,...]"""
        synapse_map = []
        projection_count = len(self.projections)
        used_rows = []
        for i in range(projection_count):
            synapse_map_part = [self.__get_proj_pop_ids(self.projections[i])]
            connections_of_p_i = self.projections[i].placed_connections
            for connections_i_j in connections_of_p_i:
                for connections_i_j_k in connections_i_j:
                    row = int(connections_i_j_k.synapse_row)
                    col = int(connections_i_j_k.synapse_on_row)
                    synapse_map_part.append([col, row])
                    if row not in used_rows:
                        used_rows.append(row)
            synapse_map.append(synapse_map_part)
        self.synapse_map = synapse_map
        unused_rows = list(range(halco.SynapseRowOnDLS.size))
        for row in used_rows:
            if row in unused_rows:
                unused_rows.remove(row)
        self.unused_rows = unused_rows
        return {"synapse_map": synapse_map, "unused_rows": unused_rows}

    def __create_pop_matrix(self):
        """
        pop matrix syntax:
        [[proj0,[pre0_index,post0_index]],
        [proj1,[pre1_index,post1_index]], ...]
        """
        pop_matrix = []
        for proj in self.projections:
            pre_pop = self.pops.index(proj.pre)
            post_pop = self.pops.index(proj.post)
            pop_matrix.append([proj, [pre_pop, post_pop]])
        self.pop_matrix = pop_matrix
        return pop_matrix

    def __create_neuron_matrix(self):
        """
        neuron matrix syntax:
        [[proj0,[pre0_index,post0_index],
            [[neuron0,neuron1],[neuron1,neuron5]]],
        [proj1,[pre1_index,post1_index],[],
            [[neuron0,neuron1],[neuron1,neuron5],[neuron2,neuron5]]]
            , ...]
        """
        neuron_matrix = []
        for element in self.pop_matrix:
            neuron_connections_in_proj = []
            pre_pop = element[1][0]
            post_pop = element[1][1]
            for connection in element[0].connections:
                pre_neuron_in_pop_index_ = connection.presynaptic_index
                post_neuron_in_pop_index_ = connection.postsynaptic_index
                pre_neuron_index = self.pops[pre_pop].all_cells[
                    pre_neuron_in_pop_index_
                ]
                post_neuron_index = self.pops[post_pop].all_cells[
                    post_neuron_in_pop_index_
                ]
                neuron_connections_in_proj.append([pre_neuron_index, post_neuron_index])
            element.append(neuron_connections_in_proj)
            neuron_matrix.append(element)
        self.neuron_matrix = neuron_matrix
        return neuron_matrix

    def __get_proj_pop_ids(self, proj: pynn.Projection):
        """
        return:
        list[presynaptic population id, postsynaptic population id]
        """
        pop_pre_index = self.pops.index(proj.pre)
        pop_post_index = self.pops.index(proj.post)
        return [pop_pre_index, pop_post_index]

    def convert_synapse_y_to_padi_coord(self, y_coord: int):
        hemisphere = math.floor(
            y_coord / (halco.SynapseRowOnDLS.size / 2)
        )  # upper or lower hemisphere
        pos_on_hemishphere = y_coord * (1 - hemisphere) + hemisphere * (
            y_coord - (halco.SynapseRowOnDLS.size / 2)
        )
        bus_per_hem = halco.PADIBusOnDLS.size / 2
        bus_size = (
            halco.SynapseDriverOnPADIBus.size * halco.SynapseRowOnSynapseDriver.size
        )  # how many rows connected to 1 bus
        bus = (
            math.floor(pos_on_hemishphere / 2) % bus_per_hem
        )  # on which bus in this hemisphere is y connected
        syndriver_on_bus = math.floor(
            pos_on_hemishphere / (bus_per_hem * 2)
        )  # to which syndriver of the bus does this row pair belong to
        pos_on_bus = syndriver_on_bus * 2 + pos_on_hemishphere % 2
        out = bus * bus_size + pos_on_bus
        out += hemisphere * (halco.SynapseRowOnDLS.size / 2)
        return out

    def __offset(self, coord: tuple, center: tuple, neuron_height: int, coord_type: Optional[str] = "xy"):
        """
        for center calculation
        """
        if coord_type == "xy":
            coord[0] += center[0] - halco.NeuronColumnOnDLS.size / 2
            coord[1] += center[1] - halco.SynapseRowOnDLS.size / 2
            coord[1] -= neuron_height
        elif coord_type == "x":
            coord += center[0] - halco.NeuronColumnOnDLS.size / 2
        elif coord_type == "y":
            coord += center[1] - halco.SynapseRowOnDLS.size / 2 - neuron_height
        return coord

    # pylint: disable = too-many-statements, too-many-branches, too-many-locals
    def __plot_chip_map(self, plot_id: int):
        """
        Given the plots index in self.plot_list, plots chipmap
        """
        plot = self.plot_list[plot_id]
        plot_type = plot["plot_type"]
        center = self.plot_list[plot_id]["settings"]["center"]
        neuron_height = self.plot_list[plot_id]["settings"]["neuron_height"]
        # neuron plotting
        for pop_id, _ in enumerate(self.neuron_map):
            first = True
            for neuron_pos in self.neuron_map[pop_id]:
                label = None
                if first:
                    label = self.pops[pop_id].label
                if plot_type is PlotTypes.SEPARATE:
                    plot["ax"][0].add_patch(
                        matplotlib.patches.Rectangle(
                            (neuron_pos[0], neuron_pos[1]),
                            1,
                            1,
                            color=self.__pop_to_hue(pop_id, plot_id),
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
                elif plot_type is PlotTypes.SANDWICH:
                    new_neuron_pos = [
                        neuron_pos[0] + center[0] - halco.NeuronColumnOnDLS.size / 2,
                        (
                            center[1]
                            - self.plot_list[plot_id]["settings"]["neuron_height"]
                            if neuron_pos[1] == 0
                            else neuron_pos[1] + center[1]
                        ),
                    ]
                    plot["ax"].add_patch(
                        matplotlib.patches.Rectangle(
                            (new_neuron_pos[0], new_neuron_pos[1]),
                            1,
                            neuron_height,
                            color=self.__pop_to_hue(pop_id, plot_id),
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
                first = False
        # synapse plotting
        projection_count = len(self.synapse_map)
        described_external_pops = []
        for i in range(projection_count):
            pops = self.synapse_map[i][0]
            connections = self.synapse_map[i][1:]
            if plot["settings"]["color_is_source"]:
                color = self.__pop_to_hue(int(pops[0]), plot_id)
            else:
                color = self.__pop_to_hue(int(pops[1]), plot_id)
            first = True
            for coord in connections:
                label = None
                if first and plot["settings"]["do_label_synapses"]:
                    label = self.projections[i].label
                elif (
                    first
                    and pops[0] not in described_external_pops
                    and pops[0] in self.ext_ids
                ):
                    label = self.pops[pops[0]].label
                    described_external_pops.append(pops[0])
                elif (
                    first
                    and pops[1] not in described_external_pops
                    and pops[1] in self.ext_ids
                ):
                    label = self.pops[pops[1]].label
                    described_external_pops.append(pops[1])
                new_coord = [coord[0], coord[1]]
                if plot["settings"]["sort_synapse_by_padi_bus"]:
                    new_coord[1] = self.convert_synapse_y_to_padi_coord(new_coord[1])
                if plot_type is PlotTypes.SEPARATE:
                    plot["ax"][1].add_patch(
                        matplotlib.patches.Rectangle(
                            (new_coord[0], new_coord[1]),
                            1,
                            1,
                            color=color,
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
                elif plot_type is PlotTypes.SANDWICH:
                    new_coord[0] += center[0]
                    new_coord[0] -= halco.NeuronColumnOnDLS.size / 2
                    if new_coord[1] >= halco.SynapseRowOnDLS.size / 2:
                        new_coord[1] += (
                            center[1]
                            + neuron_height
                            - halco.SynapseRowOnDLS.size / 2
                        )
                    else:
                        new_coord[1] += (
                            center[1]
                            - neuron_height
                            - halco.SynapseRowOnDLS.size / 2
                        )
                    plot["ax"].add_patch(
                        matplotlib.patches.Rectangle(
                            (new_coord[0], new_coord[1]),
                            1,
                            1,
                            color=color,
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
                first = False
        if plot["settings"]["highlight_empty_rows"]:
            color = (0, 0, 0, 0.2)
            first = True
            for row in self.unused_rows:
                label = None
                if first:
                    label = "unused"
                    first = False
                new_row = row + 0
                if plot["settings"]["sort_synapse_by_padi_bus"]:
                    new_row = self.convert_synapse_y_to_padi_coord(new_row)
                if plot_type is PlotTypes.SANDWICH:
                    x_coord = center[0] - halco.NeuronColumnOnDLS.size / 2
                    y_coord = (
                        center[1]
                        - halco.SynapseRowOnDLS.size / 2
                        - neuron_height
                        + new_row
                    )
                    if y_coord >= center[1] - neuron_height:
                        y_coord += neuron_height * 2
                    plot["ax"].add_patch(
                        matplotlib.patches.Rectangle(
                            (x_coord, y_coord),
                            halco.NeuronColumnOnDLS.size,
                            1,
                            color=color,
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
                elif plot_type is PlotTypes.SEPARATE:
                    plot["ax"][1].add_patch(
                        matplotlib.patches.Rectangle(
                            (0, new_row),
                            halco.NeuronColumnOnDLS.size,
                            1,
                            color=color,
                            label=label,
                            linewidth=None,
                            ec=None,
                        )
                    )
        # descriptions, labels, grids and ticks
        # grid
        if plot["settings"]["has_grid"]:
            structure_grid_color = (0, 0, 0, 0.3)
            if plot_type is PlotTypes.SANDWICH:
                # sandwich plot
                hline_anchors = [
                    [0, 0],
                    [0, halco.SynapseRowOnDLS.size / 2],
                    [0, halco.SynapseRowOnDLS.size / 2 + neuron_height],
                    [0, halco.SynapseRowOnDLS.size / 2 + neuron_height * 2],
                    [0, halco.SynapseRowOnDLS.size + neuron_height * 2],
                ]
                vline_anchors = [
                    [0, 0],
                    [halco.NeuronColumnOnDLS.size / 2, 0],
                    [halco.NeuronColumnOnDLS.size, 0],
                ]
                for hline in hline_anchors:
                    start = self.__offset(hline, center, neuron_height)
                    end = [start[0] + halco.NeuronColumnOnDLS.size, start[1]]
                    plot["ax"].hlines(
                        start[1],
                        start[0],
                        end[0],
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
                for vline in vline_anchors:
                    start = self.__offset(vline, center, neuron_height)
                    end = [
                        start[0],
                        start[1] +
                        + halco.SynapseRowOnDLS.size
                        + neuron_height * 2,
                    ]
                    plot["ax"].vlines(
                        start[0],
                        start[1],
                        end[1],
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
            elif plot_type is PlotTypes.SEPARATE:
                # separate plots
                for i in range(3):
                    # neurons
                    plot["ax"][0].hlines(
                        i,
                        0,
                        halco.NeuronColumnOnDLS.size,
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
                    plot["ax"][0].vlines(
                        (halco.NeuronColumnOnDLS.size * i) / 2,
                        0,
                        2,
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
                    # synapses
                    plot["ax"][1].hlines(
                        (halco.SynapseRowOnDLS.size * i) / 2,
                        0,
                        halco.NeuronColumnOnDLS.size,
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
                    plot["ax"][1].vlines(
                        (halco.NeuronColumnOnDLS.size * i) / 2,
                        0,
                        halco.SynapseRowOnDLS.size,
                        color=structure_grid_color,
                        linestyle="dotted",
                    )
        # gridcolor
        color = (0, 0, 0, 0.05)
        is_source_string = "Synapse color \nis target color"
        if plot["settings"]["color_is_source"]:
            is_source_string = "Synapse color \nis source color"
        # axes ticks and axis labels
        padi_int_ext_count = self.get_padibus_pop_counts()
        if plot_type is PlotTypes.SANDWICH:
            plot["ax"].set_xlim(
                center[0] - halco.NeuronColumnOnDLS.size / 2 - 16,
                center[0] + halco.NeuronColumnOnDLS.size / 2 + 16,
            )
            plot["ax"].set_ylim(
                center[1] - neuron_height - halco.NeuronColumnOnDLS.size - 16,
                center[1] + neuron_height + halco.NeuronColumnOnDLS.size + 16,
            )
            # ticks
            xticklist = [
                0,
                halco.NeuronColumnOnDLS.size / 2,
                halco.NeuronColumnOnDLS.size,
            ]
            for tick_i, _ in enumerate(xticklist):
                xticklist[tick_i] = self.__offset(
                    xticklist[tick_i], center, neuron_height, coord_type="x"
                )
            xlabels = list(map(str, xticklist))
            if plot["settings"]["sort_synapse_by_padi_bus"]:
                yticklist = [0]
                for i in range(1, 9, 1):
                    element = halco.SynapseRowOnDLS.size * (i / halco.PADIBusOnDLS.size)
                    if i <= 4:
                        yticklist.append(element)
                    if i == 4:
                        yticklist.append(element + neuron_height)
                    if i >= 4:
                        yticklist.append(element + neuron_height * 2)
                ylabels = [
                    "0",
                    "1",
                    "2",
                    "3",
                    "neuron row 1",
                    "neuron row 2",
                    "4",
                    "5",
                    "6",
                    "7",
                    "",
                ]
                for label_i in ylabels:
                    if label_i not in ["neuron row 1", "neuron row 2", ""]:
                        i = ylabels.index(label_i)
                        label_i = (
                            " bus: "
                            + label_i
                            + " "
                            + "ext rows: "
                            + str(padi_int_ext_count["external"][int(label_i)])
                            + "\n"
                            + "int rows: "
                            + str(padi_int_ext_count["onChip"][int(label_i)])
                        )
                        ylabels[i] = label_i
                plot["ax"].get_yaxis().set_tick_params(
                    which="major", labelsize=8)
                plot["ax"].set_ylabel(
                    "chip row (synapse row on PADI) ")
            else:
                # label ticks
                yticklist = [
                    0,
                    halco.SynapseRowOnDLS.size / 2,
                    halco.SynapseRowOnDLS.size / 2 + neuron_height,
                    halco.SynapseRowOnDLS.size / 2 + neuron_height * 2,
                    halco.SynapseRowOnDLS.size + neuron_height * 2,
                ]
                plot["fig"].canvas.draw()
                ylabels = list(map(str, yticklist))
                ylabels[0] = str(center[1] - halco.SynapseRowOnDLS.size / 2)
                ylabels[1] = "neuron row 0"
                ylabels[2] = "neuron row 1"
                ylabels[3] = str(center[1])
                ylabels[4] = str(center[1] + halco.SynapseRowOnDLS.size / 2)
                plot["ax"].set_ylabel("chip row")
            for tick_i, _ in enumerate(yticklist):
                yticklist[tick_i] = self.__offset(
                    yticklist[tick_i], center, neuron_height, coord_type="y"
                )
            plot["ax"].set_xticks(xticklist, xlabels)
            plot["ax"].set_yticks(yticklist, ylabels)
            box = plot["ax"].get_position()
            plot["ax"].set_position([box.x0, box.y0,
                                     box.width * 0.8, box.height])
            label = plot["ax"].get_legend_handles_labels()[1]
            ncols = math.floor(len(label) / 13) + 1
            plot["ax"].legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title=is_source_string,
                ncols=ncols,
            )
            # finalise
            plot["ax"].set_xlabel("chip column")
            title = "synapse and neuron placement on chip"
            if plot["settings"]["custom_title"] is not None:
                title = plot["settings"]["custom_title"]
            plot["fig"].suptitle(title)
            # grid
            if plot["settings"]["has_grid"]:
                minor_ticks_x = np.arange(
                    xticklist[0], xticklist[len(xticklist) - 1], 1
                )
                minor_ticks_y = np.arange(
                    yticklist[0], yticklist[len(yticklist) - 1], 1
                )
                plot["ax"].set_xticks(minor_ticks_x, minor=True)
                plot["ax"].set_yticks(minor_ticks_y, minor=True)
                plot["ax"].grid(
                    which="both", linewidth=0.05,
                    color=color, linestyle="dotted"
                )
        elif plot_type is PlotTypes.SEPARATE:
            plot["ax"][0].set_xlim(0 - 16, halco.NeuronColumnOnDLS.size + 16)
            plot["ax"][0].set_ylim(0, 2)
            plot["ax"][1].set_ylim(0 - 16, halco.SynapseRowOnDLS.size
                                   * (i / 8) + 16)
            # separate plot
            x1ticklist = [
                0,
                halco.NeuronColumnOnDLS.size / 2,
                halco.NeuronColumnOnDLS.size,
            ]
            y1ticklist = [0, 1, 2]
            y2ticklist = [0, halco.SynapseRowOnDLS.size / 2,
                          halco.SynapseRowOnDLS.size]
            if plot["settings"]["sort_synapse_by_padi_bus"]:
                y2ticklist = [halco.SynapseRowOnDLS.size
                              * (i / 8) for i in range(9)]
                y2labels = [str(x) for x, _ in enumerate(y2ticklist)]
                y2labels[len(y2labels) - 1] = ""
                for label_i in y2labels:
                    if label_i not in ["neuron row 1", "neuron row 2", ""]:
                        i = y2labels.index(label_i)
                        label_i = (
                            " bus: "
                            + label_i
                            + " "
                            + "ext rows: "
                            + str(padi_int_ext_count["external"][int(label_i)])
                            + "\n"
                            + "int rows: "
                            + str(padi_int_ext_count["onChip"][int(label_i)])
                        )
                        y2labels[i] = label_i
                plot["ax"][1].get_yaxis().set_tick_params(which="major",
                                                          labelsize=5)
                plot["ax"][1].set_ylabel("Synapse row on PADI Bus")
                plot["ax"][1].set_yticks(y2ticklist, y2labels)
            else:
                plot["ax"][1].set_ylabel("synapse row")
                plot["ax"][1].set_yticks(y2ticklist)
            plot["ax"][0].set_xticks(x1ticklist)
            plot["ax"][0].set_yticks(y1ticklist)
            plot["ax"][1].set_xticks(x1ticklist)
            plot["ax"][0].set_xlabel("neuron column")
            plot["ax"][1].set_xlabel("synapse column")
            plot["ax"][0].set_ylabel("neuron row")
            title1 = "neuron placement on chip"
            title2 = "synapse placement on chip"
            if plot["settings"]["custom_title"] is not None:
                if (
                    isinstance(plot["settings"]["custom_title"], list)
                    and len(plot["settings"]["custom_title"]) == 2
                ):
                    title1 = plot["settings"]["custom_title"][0]
                    title2 = plot["settings"]["custom_title"][1]
                elif isinstance(plot["settings"]["custom_title"], str):
                    title1 = plot["settings"]["custom_title"]
                    title2 = plot["settings"]["custom_title"]
            plot["ax"][0].title.set_text(title1)
            plot["ax"][1].title.set_text(title2)
            lines = []
            labels = []
            for ax in plot["fig"].axes:
                line, label = ax.get_legend_handles_labels()
                lines.extend(line)
                labels.extend(label)
            labels_current = plot["ax"][0].get_legend_handles_labels()[1]
            ncols = math.floor(len(labels_current) / 13) + 1
            plot["fig"].legend(
                lines,
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title=is_source_string,
                ncols=ncols,
            )
            plot["fig"].tight_layout()
            # grid
            if plot["settings"]["has_grid"]:
                minor_ticks_x = np.arange(
                    x1ticklist[0], x1ticklist[len(x1ticklist) - 1], 1
                )
                minor_ticks_y1 = np.arange(
                    y1ticklist[0], y1ticklist[len(y1ticklist) - 1], 1
                )
                minor_ticks_y2 = np.arange(
                    y2ticklist[0], y2ticklist[len(y2ticklist) - 1], 1
                )
                plot["ax"][1].set_xticks(minor_ticks_x, minor=True)
                plot["ax"][0].set_yticks(minor_ticks_y1, minor=True)
                plot["ax"][1].set_yticks(minor_ticks_y2, minor=True)
                plot["ax"][0].grid(
                    which="both", linewidth=0.05,
                    color=color, linestyle="dotted"
                )
                plot["ax"][1].grid(
                    which="both", linewidth=0.05,
                    color=color, linestyle="dotted"
                )

    def __plot_pop_matrix(self, plot_id: int):
        """
        Given the plots index in self.plot_list, plots population matrix plot
        """
        already_labelled = []
        for element in self.pop_matrix:
            # element [proj_i,[popj_index,popk_index]]
            if self.plot_list[plot_id]["settings"]["color_is_source"]:
                color = self.__pop_to_hue(element[1][0], plot_id)
                label = (
                    None
                    if self.pops[element[1][0]].label in already_labelled
                    else self.pops[element[1][0]].label
                )
                already_labelled.append(label)
            else:
                color = self.__pop_to_hue(element[1][1], plot_id)
                label = (
                    None
                    if self.pops[element[1][1]].label in already_labelled
                    else self.pops[element[1][1]].label
                )
                already_labelled.append(label)
            if self.plot_list[plot_id]["settings"]["do_label_synapses"]:
                label = element[0].label
            self.plot_list[plot_id]["ax"].add_patch(
                matplotlib.patches.Rectangle(
                    (element[1][0], element[1][1]),
                    1,
                    1,
                    color=color,
                    label=label,
                    linewidth=None,
                    ec=None,
                )
            )
        # axes grids labels titles
        color = (0, 0, 0, 0.05)
        is_source_string = "Synapse color \nis target color"
        if self.plot_list[plot_id]["settings"]["color_is_source"]:
            is_source_string = "Synapse color \nis source color"
        self.plot_list[plot_id]["ax"].set_xlabel("source population")
        self.plot_list[plot_id]["ax"].set_ylabel("target population")
        ticks = np.arange(len(self.pops) + 1)
        labels_current = self.plot_list[plot_id]["ax"].get_legend_handles_labels()[1]
        ncols = math.floor(len(labels_current) / 13) + 1
        self.plot_list[plot_id]["fig"].legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=is_source_string,
            ncols=ncols,
        )
        self.plot_list[plot_id]["ax"].set_xticks(ticks)
        self.plot_list[plot_id]["ax"].set_yticks(ticks)
        max_xy = [0, 0]
        max_xy[0] = max(element[1][0] for element in self.pop_matrix)
        max_xy[1] = max(element[1][1] for element in self.pop_matrix)
        self.plot_list[plot_id]["ax"].set_xlim(0, max_xy[0] + 1)
        self.plot_list[plot_id]["ax"].set_ylim(0, max_xy[1] + 1)
        self.plot_list[plot_id]["fig"].tight_layout()
        title = "Connections between populations"
        if self.plot_list[plot_id]["settings"]["custom_title"] is not None:
            title = self.plot_list[plot_id]["settings"]["custom_title"]
        self.plot_list[plot_id]["ax"].title.set_text(title)
        if self.plot_list[plot_id]["settings"]["has_grid"]:
            self.plot_list[plot_id]["ax"].grid(
                linewidth=0.1, color=color, linestyle="dotted"
            )

    # pylint: disable = too-many-statements, too-many-branches, too-many-locals
    def __plot_neuron_matrix(self, plot_id: int):
        """
        Given the plots index in self.plot_list, plots neuron matrix plot
        """
        already_labelled = []
        max_xy = [0, 0]
        for element in self.neuron_matrix:
            # element [[proj0,[pre_pop_index,post_pop_index],
            # [[neuron0_index,neuron1_index],[neuron1_index,neuron5_index]]]
            if self.plot_list[plot_id]["settings"]["color_is_source"]:
                color = self.__pop_to_hue(element[1][0], plot_id)
                label = (
                    None
                    if self.pops[element[1][0]] in already_labelled
                    else self.pops[element[1][0]].label
                )
                if label is not None:
                    already_labelled.append(self.pops[element[1][0]])
            else:
                color = self.__pop_to_hue(element[1][1], plot_id)
                label = (
                    None
                    if self.pops[element[1][1]] in already_labelled
                    else self.pops[element[1][1]].label
                )
                if label is not None:
                    already_labelled.append(self.pops[element[1][1]])
            if self.plot_list[plot_id]["settings"]["do_label_synapses"]:
                label = element[0].label
            first = True
            for neuron_pair in element[2]:
                if not first:
                    label = None
                first = False
                if neuron_pair[0] > max_xy[0]:
                    max_xy[0] = neuron_pair[0]
                if neuron_pair[1] > max_xy[1]:
                    max_xy[1] = neuron_pair[1]
                self.plot_list[plot_id]["ax"].add_patch(
                    matplotlib.patches.Rectangle(
                        (neuron_pair[0], neuron_pair[1]),
                        1,
                        1,
                        color=color,
                        label=label,
                        linewidth=None,
                        ec=None,
                    )
                )
        # axes grids labels titles
        color = (0, 0, 0, 0.05)
        is_source_string = "Synapse color \nis target color"
        if self.plot_list[plot_id]["settings"]["color_is_source"]:
            is_source_string = "Synapse color \nis source color"
        self.plot_list[plot_id]["ax"].set_xlabel("source neuron id")
        self.plot_list[plot_id]["ax"].set_ylabel("target neuron id")
        if max_xy[0] <= halco.SynapseRowOnDLS.size:
            max_xy = [halco.SynapseRowOnDLS.size - 1, max_xy[1]]
        if max_xy[1] <= halco.SynapseRowOnDLS.size:
            max_xy = [max_xy[0], halco.SynapseRowOnDLS.size - 1]
        self.plot_list[plot_id]["ax"].set_xlim(0 - 16, max_xy[0] + 1 + 16)
        self.plot_list[plot_id]["ax"].set_ylim(0 - 16, max_xy[1] + 1 + 16)
        xticks = np.arange(
            0,
            max([max_xy[0], halco.SynapseRowOnDLS.size + 1]),
            step=halco.SynapseRowOnDLS.size / 4,
        )
        yticks = np.arange(
            0,
            max([max_xy[1], halco.SynapseRowOnDLS.size + 1]),
            step=halco.SynapseRowOnDLS.size / 4,
        )
        minor_xticks = np.arange(
            0, max([max_xy[0], halco.SynapseRowOnDLS.size + 1]), step=16
        )
        minor_yticks = np.arange(
            0, max([max_xy[1], halco.SynapseRowOnDLS.size + 1]), step=16
        )
        self.plot_list[plot_id]["ax"].set_xticks(minor_xticks, minor=True)
        self.plot_list[plot_id]["ax"].set_yticks(minor_yticks, minor=True)
        labels_current = self.plot_list[plot_id]["ax"].get_legend_handles_labels()[1]
        ncols = math.floor(len(labels_current) / 13) + 1
        self.plot_list[plot_id]["fig"].legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=is_source_string,
            ncols=ncols,
        )
        self.plot_list[plot_id]["ax"].set_xticks(xticks)
        self.plot_list[plot_id]["ax"].set_yticks(yticks)
        self.plot_list[plot_id]["fig"].tight_layout()
        title = "Connections between neurons"
        if self.plot_list[plot_id]["settings"]["custom_title"] is not None:
            title = self.plot_list[plot_id]["settings"]["custom_title"]
        self.plot_list[plot_id]["ax"].title.set_text(title)
        if self.plot_list[plot_id]["settings"]["has_grid"]:
            structure_grid_color = (0, 0, 0, 0.3)
            minor_ticks_x = np.arange(xticks[0], xticks[len(xticks) - 1], 1)
            minor_ticks_y = np.arange(yticks[0], yticks[len(yticks) - 1], 1)
            self.plot_list[plot_id]["ax"].set_xticks(minor_ticks_x, minor=True)
            self.plot_list[plot_id]["ax"].set_yticks(minor_ticks_y, minor=True)
            self.plot_list[plot_id]["ax"].grid(
                which="both",
                linewidth=0.05,
                color=structure_grid_color,
                linestyle="dotted",
            )

    def plot_all(self):
        """
        plot all plots in plot_list
        """
        self.retrieve_data(True, True, True, True)
        for i, _ in enumerate(self.plot_list):
            self.__plot_by_id(i, False)

    def __plot_by_id(self, plot_id: int, do_retrieve_data: bool):
        """
        plot plot_list[id]
        """
        plot = self.plot_list[plot_id]
        if plot["plot_type"] in [PlotTypes.SANDWICH, PlotTypes.SEPARATE]:
            if do_retrieve_data:
                self.retrieve_data(True, True, False, False)
            self.__plot_chip_map(plot_id)
        elif plot["plot_type"] is PlotTypes.POPULATIONMATRIX:
            if do_retrieve_data:
                self.retrieve_data(False, True, True, False)
            self.__plot_pop_matrix(plot_id)
        elif plot["plot_type"] is PlotTypes.NEURONMATRIX:
            if do_retrieve_data:
                self.retrieve_data(False, True, True, True)
            self.__plot_neuron_matrix(plot_id)

    def plot_by_id(self, plot_id: int):
        """
        plot plot_list[id]
        """
        self.__plot_by_id(plot_id, True)

    # pylint: disable = too-many-locals
    def __pop_to_hue(self, pop_id: int, plot_id: int):
        """
        converts population id to hue for plotting
        """
        if (
            self.plot_list[plot_id]["settings"]["population_contrast_groups"]
            is not None
        ):
            contrast_groups = self.plot_list[plot_id]["settings"][
                "population_contrast_groups"
            ]
            group_count = len(contrast_groups)
            counter = 0
            hue = 0
            saturation = 0
            value = 0
            names = contrast_groups.keys()
            for name in names:
                group = contrast_groups[name]
                if pop_id in group:
                    h_offset = counter / group_count
                    index = group.index(pop_id)
                    lengroup = 1
                    if len(group) > 1:
                        lengroup = len(group) - 1
                    hue = (
                        h_offset
                        + (index / (lengroup)) * (2 / 3) * (1 / group_count)
                        + 200 / 360
                    )
                    smin = 0.3
                    saturation = smin + index * (1 - smin) / lengroup
                    vmin = 0.8
                    value = 1 - index * (1 - vmin) / lengroup
                counter += 1
            rgb_c = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb = [
                rgb_c[0],
                rgb_c[1],
                rgb_c[2],
                self.plot_list[plot_id]["settings"]["alpha"],
            ]
        else:
            angle = (pop_id / len(self.pops) * 2 * np.pi) / 2
            red = np.sin(angle)
            green = np.sin(angle - np.pi / 3)
            blue = np.sin(angle - 2 * np.pi / 3)
            rgb = [
                float(np.abs(red)),
                float(np.abs(green)),
                float(np.abs(blue)),
                self.plot_list[plot_id]["settings"]["alpha"],
            ]
        return rgb

    def save_all(self, folder: Optional[str] = "",
                 name: Optional[str] = "",
                 formats=None, dpi=900):
        """
        save all plots

        Args:
        :param folder: (string) folder
        :param name: (string) file name
        :param formats: (list<string>) formats to be saved as, e.g. ['png', 'pdf']
        :param dpi: (int) resolution for png or jpeg
        """
        if formats is None:
            formats = ["png", "pdf"]
        for i, _ in enumerate(self.plot_list):
            self.save_by_id(
                plot_id=i,
                folder=folder,
                name=name,
                formats=formats,
                dpi=dpi,
                sub_name=str(i),
            )

    def save_by_id(
            self, plot_id,
            folder: Optional[str] = "",
            name: Optional[str] = "",
            formats: Optional[list] = None,
            dpi: Optional[int] = 900,
            sub_name: Optional[str] = ""):
        """
        saves one plot

        Args:
        :param plot_id: (int) id of plot in plot_list to be saved
        :param folder: (string) folder
        :param name: (string) file name
        :param sub_name: (string) optional name behind name
        :param formats: (list<string>) formats to be saved as, e.g. ['png', 'pdf']
        :param dpi: (int) resolution for png or jpeg
        """
        if formats is None:
            formats = ["png", "pdf"]
        plot = self.plot_list[plot_id]
        plot_type = plot["plot_type"]
        if plot_type is PlotTypes.SANDWICH:
            name = name if name != "" else "sandwich"
        elif plot_type is PlotTypes.SEPARATE:
            name = name if name != "" else "separate"
        elif plot_type is PlotTypes.POPULATIONMATRIX:
            name = name if name != "" else "pop_matrix"
        elif plot_type is PlotTypes.NEURONMATRIX:
            name = name if name != "" else "neuron_matrix"
        self.__save(
            plot_id=plot_id,
            folder=folder,
            name=name + sub_name,
            formats=formats,
            dpi=dpi,
        )

    def __save(
            self, plot_id: int,
            folder: Optional[str] = "",
            name: Optional[str] = "",
            formats: Optional[list] = None,
            dpi: Optional[int] = 900):
        '''
        saves one plot
        '''
        if not os.path.exists(folder):
            os.makedirs(folder)
        if formats is None:
            formats = ["png", "pdf"]
        if "png" in formats:
            self.plot_list[plot_id]["fig"].savefig(
                folder + name + ".png", bbox_inches="tight", dpi=dpi
            )
        if "pdf" in formats:
            self.plot_list[plot_id]["fig"].savefig(
                folder + name + ".pdf", bbox_inches="tight"
            )

    def remove_plot(self, plot_id: int):
        if len(self.plot_list) > plot_id >= 0:
            self.plot_list.pop(plot_id)

    def modify_plot(self, plot_id: int, setting_name: str, new_setting_value):
        self.plot_list[plot_id]["settings"][setting_name] = new_setting_value

    def clear_plot(self, plot_id: int):
        plot_type = self.plot_list[plot_id]["plot_type"]
        if plot_type == PlotTypes.SEPARATE:
            fig = plt.figure()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
            ax = [ax1, ax2]
        else:
            fig, ax = plt.subplots()
        old_fig = self.plot_list[plot_id]["fig"]
        self.plot_list[plot_id]["fig"] = fig
        self.plot_list[plot_id]["ax"] = ax
        plt.close(old_fig)

    def clear_all_plots(self):
        for plot_id, _ in enumerate(self.plot_list):
            self.clear_plot(plot_id)

    def get_padibus_pop_counts(self):
        """
        return: {'external'[padibus0 external row count, padibus1...]:,
        'onChip':[padibus0 internal row count, padibus1...]}
        """
        row_types = ["external", "onChip"]
        padi_used_rows = {}
        for row_type in row_types:
            padi_used_rows[row_type] = [0 for _ in range(8)]
        already_y = []
        for proj in self.synapse_map:
            source_pop_ids = proj[0][0]
            row_type = (
                "external"
                if source_pop_ids in [element[0] for element in self.external_pops_id]
                else "onChip"
            )
            row_list = [s[1] for s in proj[1: len(proj)]]
            for y_coord in row_list:
                if y_coord not in already_y:
                    y_in_padi_coord = self.convert_synapse_y_to_padi_coord(y_coord)
                    bus = math.floor(
                        y_in_padi_coord
                        / (
                            halco.SynapseDriverOnPADIBus.size
                            * halco.SynapseRowOnSynapseDriver.size
                        )
                    )
                    padi_used_rows[row_type][bus] += 1
                    already_y.append(y_coord)
        return padi_used_rows

    def get_most_often_postneuron(self):
        """
        returns index of pop of most targetted,
        neuron on neuron matrix,
        most targetted neuron on neuron matrix
        and how many times it is targetted
        """
        # syntax: {pop_id:[neur0 = targettedcount,
        # ..., neurpop.size = targettedcount]}
        neuron_target_count = {}
        for pop_id, _ in enumerate(self.pops):
            starting_dict = {}
            for neuron_id in range(self.pops[pop_id].size):
                starting_dict[neuron_id] = 0
            neuron_target_count[pop_id] = starting_dict
        for proj in self.projections:
            post_id = self.pops.index(proj.post)
            for connection in proj.connections:
                neuron_target_count[post_id][connection.postsynaptic_index] += 1
        # determine max
        postneuron_id = -1
        post_id = -1
        max_targetted_count = -1
        # neuron target count structure
        # [[pop0 neuron 0 target count,
        # pop0 neuron1 target count, ...],[pop1...],...]
        for pop_id, postpop in enumerate(neuron_target_count):
            for neuron_id, neuron_count in enumerate(postpop):
                if neuron_count > max_targetted_count:
                    max_targetted_count = neuron_count
                    post_id = pop_id
                    postneuron_id = neuron_id
        return post_id, postneuron_id, max_targetted_count
