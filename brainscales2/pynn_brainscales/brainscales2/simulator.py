# pylint: disable=too-many-lines

from typing import ClassVar
import numpy as np
from pyNN.common import IDMixin, Population, Connection
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells import HXNeuron, \
    SpikeSourceArray
from dlens_vx_v1 import hal, halco, sta, hxcomm, lola, logger


name = "HX"  # for use in annotating output data


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        int.__init__(n)
        IDMixin.__init__(self)


class ConnectionConfigurationBuilder:
    """
    Builder Pattern to generate coord-config pairs for the synapse drivers
    and the synapse matrices.
    """

    def __init__(self):
        # lists of coord-config pairs
        self._synapse_drivers = np.ndarray(shape=(0, 2))
        self._synapse_matrices = np.ndarray(shape=(0, 2))

        # list of allocated synapse drivers
        self._used_syndrv = np.zeros(0, dtype={
            "names": ("synapse_driver", "row", "pre_population",
                      "post_populations", "receptor_type"),
            "formats": (halco.SynapseDriverOnDLS, "i1", "i2", list, "U10")})

        # list of PADI busses with number of used input rows - 1
        self._used_padi_rows = np.ndarray(shape=(0, 2))

        # list of external connections [pre, post, weight, spiketimes]
        self._external_connections = np.zeros(0, dtype={
            "names": ("pre", "post", "weight", "receptor_type", "spiketimes"),
            "formats": ("i2", halco.AtomicNeuronOnDLS, "i1", "U10", list)})

        # list of external events
        self._external_events = np.zeros(0, dtype={
            "names": ("spiketimes", "label"),
            "formats": (object, "u2")})

    def generate(self) -> [sta.PlaybackProgramBuilder, np.ndarray]:
        """
        Generate a builder with configured synapse drivers and synapse matix
        and also return a list of external events to be injected.
        """
        builder = sta.PlaybackProgramBuilder()
        self._add_external()
        for coord, config in self._synapse_drivers:
            builder.write(coord, config)
        for coord, config in self._synapse_matrices:
            builder.write(coord, config)

        external_events = self._sort_external_events()

        return builder, external_events

    def _sort_external_events(self):
        input_spikes = np.zeros(shape=(0, 2))
        for spiketimes, label in self._external_events:
            for time in spiketimes:
                input_spikes = np.append(input_spikes, np.array([[time,
                                                                  label]]),
                                         axis=0)
        input_spikes = input_spikes[input_spikes[:, 0].argsort()]
        return input_spikes

    def add(self, connection: Connection) -> None:
        """
        Add the given connection. On-chip connections are configured directly,
        while external connections are added after all on-chip connections,
        due to their larger variability.
        """

        # firstly on-chip connections (currently only celltype 'HXNeuron') are
        # configured
        if isinstance(connection.presynaptic_cell.celltype, HXNeuron):
            # synapse drivers for on-chip connections are allocated starting
            # with the smallest index
            syndrv_coord = self._synapse_driver_coord(connection)
            # drivers receive all inputs, will be updated if external events
            # are used in the program, so a driver will only forward either
            # on-chip or external input
            mask = 0b00000
            self._configure_synapse_drivers(syndrv_coord, mask)

            # same for the synapse matrix
            synmtx_coord = self._synapse_matrix_coord(connection)
            config, coord_index = self._find_used_synapse_matrix(synmtx_coord)
            synmtx_config = self._synapse_matrix_config(config,
                                                        connection,
                                                        syndrv_coord)
            self._update_synapse_matrix(coord_index,
                                        synmtx_coord,
                                        synmtx_config)

        # external connections are stored in a list and added after all on-chip
        # connections are configured
        elif isinstance(connection.presynaptic_cell.celltype,
                        SpikeSourceArray):
            pre = connection.presynaptic_cell
            post = halco.AtomicNeuronOnDLS(halco.Enum(
                state.neuron_placement[connection.postsynaptic_cell]))
            weight = connection.weight
            receptor_type = connection.projection.receptor_type
            spiketimes = pre.celltype.parameter_space["spike_times"] \
                .base_value.value
            external_connection = np.array(
                [(pre, post, weight, receptor_type, spiketimes)],
                dtype=self._external_connections.dtype)
            self._external_connections = np.append(self._external_connections,
                                                   external_connection)

        # other celltypes are not supported yet
        else:
            raise NotImplementedError

    def _synapse_driver_coord(self, connection: Connection) -> np.ndarray:
        """
        Calculate the synapse driver coordinate.
        """

        pre = state.neuron_placement[connection.presynaptic_cell]
        post = state.neuron_placement[connection.postsynaptic_cell]

        # check if a synapse driver for the presynaptic neuron with given
        # receptor type was already allocated and if its synapse to the
        # postsynaptic neuron is still available
        coord_index = np.where(
            (self._used_syndrv["pre_population"] == pre)
            & (post not in self._used_syndrv["post_populations"]))
        coord_index = coord_index[0]
        for coord in coord_index:
            if self._used_syndrv[coord]["receptor_type"] \
                    == connection.projection.receptor_type:
                self._used_syndrv[coord]["post_populations"].append(post)
                return self._used_syndrv[coord]

        # calculate crossbar output, PADI bus
        # neurons per crossbar input channel
        neurons = int(halco.NeuronColumnOnDLS.size
                      / halco.NeuronEventOutputOnDLS.size)
        crossbar_output = halco.CrossbarOutputOnDLS(
            pre // neurons
            - (pre // (halco.NeuronEventOutputOnNeuronBackendBlock.size
                       * neurons)) * halco.PADIBusOnPADIBusBlock.size
            + (pre // (halco.PADIBusOnDLS.size * neurons))
            * halco.PADIBusOnPADIBusBlock.size)
        padi_bus = crossbar_output.toPADIBusOnDLS()

        # check if there is a driver available
        padibus_index, _ = np.where(self._used_padi_rows == padi_bus)
        if len(padibus_index) == 0:
            padi_row_index = 0
            self._used_padi_rows = np.append(self._used_padi_rows,
                                             [[padi_bus, 0]], axis=0)
        elif self._used_padi_rows[int(padibus_index)][1] \
                >= lola.SynapseMatrix.Weight.max:
            raise RuntimeError("""Too many connections. Try decreasing the
                               weight values in a way they do not exceed large
                               multiples of the maximum synaptic weight (63) or
                               reduce the number of projections with the same
                               presynaptic population, but different receptor
                               types.""")
        else:
            padi_row_index = self._used_padi_rows[int(padibus_index)][1] + 1
            self._used_padi_rows[padi_bus][1] = padi_row_index

        # set synapse driver
        syndrv = halco.SynapseDriverOnSynapseDriverBlock(
            int(halco.SynapseDriverOnPADIBus(padi_row_index // 2))
            * halco.PADIBusOnPADIBusBlock.size
            + int(padi_bus.toPADIBusOnPADIBusBlock()))
        global_syndrv = halco.SynapseDriverOnDLS(
            syndrv, padi_bus.toPADIBusBlockOnDLS().toSynapseDriverBlockOnDLS())

        used_syndrv = np.array([(global_syndrv, padi_row_index % 2, pre,
                                 [post], connection.projection.receptor_type)],
                               dtype=self._used_syndrv.dtype)
        self._used_syndrv = np.append(self._used_syndrv, used_syndrv)

        return used_syndrv

    def _configure_synapse_drivers(self, syndrv_coord: np.ndarray, mask: int) \
            -> None:
        """
        Create a new synapse driver configuration or update the already
        existing one.
        """

        # check if there already is a synapse driver config for this coord
        synapse_drivers_coords = self._synapse_drivers[:, 0]
        coord_index = np.where(
            synapse_drivers_coords == syndrv_coord["synapse_driver"])
        coord_index = coord_index[0]

        # if not a new one is created
        if len(coord_index) == 0:
            config = hal.SynapseDriverConfig()
            config.enable_receiver = True
            config.enable_address_out = True
            config.row_address_compare_mask = mask
        # if there is, it is used and extended
        else:
            assert len(coord_index) == 1
            config = self._synapse_drivers[int(coord_index)][1]
        syndrv_config = self._synapse_driver_config(config, syndrv_coord)

        # if a new config was created, it is appended to the list of pairs
        if len(coord_index) == 0:
            self._synapse_drivers = \
                np.append(self._synapse_drivers,
                          [[syndrv_coord["synapse_driver"][0], syndrv_config]],
                          axis=0)
        # otherwise the entry is overwritten
        else:
            if isinstance(syndrv_coord["synapse_driver"], np.ndarray):
                syndrv = syndrv_coord["synapse_driver"][0]
            elif isinstance(syndrv_coord["synapse_driver"],
                            halco.SynapseDriverOnDLS):
                syndrv = syndrv_coord["synapse_driver"]
            else:
                raise NotImplementedError
            self._synapse_drivers[int(coord_index)] = \
                [syndrv, syndrv_config]

    @staticmethod
    def _synapse_driver_config(drv_config: hal.SynapseDriverConfig,
                               drv_coord: np.ndarray) \
            -> hal.SynapseDriverConfig:
        """
        Configure the receptor type of the row to use.
        """

        receptor_type = drv_coord["receptor_type"]
        row = drv_coord["row"]
        assert receptor_type in ["excitatory", "inhibitory"]
        assert row in [0, 1]

        if row == 0 and receptor_type == "inhibitory":
            drv_config.row_mode_bottom = drv_config.RowMode.inhibitory

        elif row == 0 and receptor_type == "excitatory":
            drv_config.row_mode_bottom = drv_config.RowMode.excitatory

        elif row == 1 and receptor_type == "inhibitory":
            drv_config.row_mode_top = drv_config.RowMode.inhibitory

        else:
            drv_config.row_mode_top = drv_config.RowMode.excitatory

        return drv_config

    @staticmethod
    def _synapse_matrix_coord(connection: Connection) -> halco.SynramOnDLS:
        """
        Determine synapse matrix coordinate.
        """

        post = state.neuron_placement[connection.postsynaptic_cell]
        nrn = halco.AtomicNeuronOnDLS(halco.Enum(post))
        return nrn.toNeuronRowOnDLS().toSynramOnDLS()

    def _find_used_synapse_matrix(self, synmtx_coord: halco.SynramOnDLS) \
            -> [lola.SynapseMatrix, np.ndarray]:
        """
        Return the already existing synapse matrix configuration for the given
        coordinate or create a new one.
        """

        # check if there already is a config for the synapse matrix at the
        # given coordinate (hemisphere)
        coord_index, _ = np.where(self._synapse_matrices == synmtx_coord)
        # if not, create a new one
        if len(coord_index) == 0:
            synmtx_config = lola.SynapseMatrix()
        # otherwise take the existing one
        else:
            assert len(coord_index) == 1
            synmtx_config = self._synapse_matrices[int(coord_index)][1]
        return synmtx_config, coord_index

    @staticmethod
    def _synapse_matrix_config(synapse_matrix: lola.SynapseMatrix,
                               connection: Connection,
                               syndrv_coord: np.ndarray) -> lola.SynapseMatrix:
        """
        Configure the label and weight for the given connection.
        """

        pre = int(state.neuron_placement[connection.presynaptic_cell])
        pre = halco.AtomicNeuronOnDLS(halco.Enum(pre))
        post = int(state.neuron_placement[connection.postsynaptic_cell])
        # neurons per crossbar input channel
        neurons = int(halco.NeuronColumnOnDLS.size
                      / halco.NeuronEventOutputOnDLS.size)
        addr = (pre.toNeuronColumnOnDLS().toEnum() % neurons) \
            + (pre.toNeuronRowOnDLS().toEnum() * neurons)
        syndrv_coord["synapse_driver"] = np.atleast_1d(
            syndrv_coord["synapse_driver"])
        pre_coord = halco.SynapseRowOnSynapseDriver.size \
            * int(syndrv_coord["synapse_driver"][0].
                  toSynapseDriverOnSynapseDriverBlock().toEnum()) \
            + syndrv_coord["row"]
        pre_coord = int(pre_coord)
        synapse_matrix.labels[pre_coord][post] = int(addr)
        synapse_matrix.weights[pre_coord][post] = int(connection.weight)
        return synapse_matrix

    def _update_synapse_matrix(self,
                               coord_index: np.ndarray,
                               synmtx_coord: halco.SynramOnDLS,
                               synmtx_config: lola.SynapseMatrix) -> None:
        """
        Update the entry in self._synapse_matrices for the given coordinate or
        create a new one, if it does not exist yet.
        """

        if len(coord_index) == 0:
            self._synapse_matrices = np.append(
                self._synapse_matrices,
                [[synmtx_coord, synmtx_config]],
                axis=0)
        else:
            self._synapse_matrices[int(coord_index)] = \
                [synmtx_coord, synmtx_config]

    def _add_external(self) -> None:
        """
        Add the connections from self._external_connections.
        """
        # determine the unused synapse drivers and update the compare masks
        # of those forwarding on-chip events
        free_padibusses, free_rows_top, free_rows_bottom = \
            self._find_unused_synapse_drivers()
        self._update_compare_mask(free_padibusses,
                                  [len(free_rows_top), len(free_rows_bottom)])

        # make a list of all presynaptic sources
        pres = np.unique(self._external_connections["pre"])
        for hemisphere in halco.iter_all(halco.HemisphereOnDLS):
            # memorize used matrix entries
            used_entries = np.zeros(0, dtype={
                "names": ("bus", "row", "posts", "number_pres"),
                "formats": (halco.PADIBusOnPADIBusBlock, "i1", object, "i2")})

            for receptor_type in ["excitatory", "inhibitory"]:
                # memorize all connections with the same presynaptic source
                connections = np.ndarray(shape=(0, 2))
                for ind_pre, val_pre in enumerate(pres):
                    indices = np.where(
                        (self._external_connections["pre"] == val_pre)
                        & (self._external_connections["receptor_type"]
                           == receptor_type)
                        & (self._external_connections["post"][0].
                           toNeuronRowOnDLS().toHemisphereOnDLS()
                           == hemisphere))[0]
                    if len(indices) > 0:
                        connections = np.append(
                            connections,
                            [[ind_pre, self._external_connections[indices]]],
                            axis=0)

                # TODO: don't return something
                # configure synapse matrices
                used_entries = self._external_configuration(
                    connections, free_rows_top, used_entries, hemisphere)

    def _find_unused_synapse_drivers(self) \
            -> [np.ndarray, np.ndarray, np.ndarray]:
        """
        Check which PADI bus has how many unused synapse driver and make
        lists with free synapse driver rows and their padibusses per
        hemisphere.
        """

        free_padibusses = np.ndarray(shape=(0, 2))
        free_rows_top = np.zeros(0, dtype={
            "names": ("padibus", "syndrv_row"),
            "formats": (halco.PADIBusOnPADIBusBlock, "i1")})
        free_rows_bottom = np.zeros(0, dtype={
            "names": ("padibus", "syndrv_row"),
            "formats": (halco.PADIBusOnPADIBusBlock, "i1")})

        for padibus in halco.iter_all(halco.PADIBusOnDLS):
            used, _ = np.where(self._used_padi_rows == padibus)
            if len(used) == 0:
                free_rows = 64
                mask = 0b11111
            else:
                used_rows = self._used_padi_rows[used][0][1] + 1
                if used_rows <= 4:
                    free_rows = 56
                    mask = 0b11100
                elif used_rows <= 8:
                    free_rows = 48
                    mask = 0b11000
                elif used_rows <= 16:
                    free_rows = 32
                    mask = 0b10000
                else:
                    free_rows = 0
                    mask = 0b00000

            free_padibusses = np.append(free_padibusses,
                                        [[padibus, mask]],
                                        axis=0)

            start_syndrv = (64 - free_rows) / 2
            entry = [None] * free_rows
            for row in range(free_rows):
                entry[row] = (padibus.toPADIBusOnPADIBusBlock(),
                              start_syndrv + row)
            if padibus.toPADIBusBlockOnDLS() == 0:
                free_rows_top = np.append(
                    free_rows_top, np.array(entry, dtype=free_rows_top.dtype))
            else:
                free_rows_bottom = np.append(
                    free_rows_bottom,
                    np.array(entry, dtype=free_rows_bottom.dtype))

        return free_padibusses, free_rows_top, free_rows_bottom

    def _update_compare_mask(self,
                             free_padibusses: np.ndarray,
                             free_rows: list) -> None:
        """
        Update the compare_mask of the synapse drivers forwarding on-chip
        events in a way, so that they do not forward external events.
        """

        # check whether the maximal number of presynaptic sources connected to
        # one postsynaptic neuron exceeds the number of available drivers for
        # each hemisphere
        for hemisphere in halco.iter_all(halco.HemisphereOnDLS):
            post_exc = np.array(
                [post for post in self._external_connections[
                    self._external_connections["receptor_type"]
                    == "excitatory"]["post"]
                 if post.toNeuronRowOnDLS().toHemisphereOnDLS() == hemisphere])

            post_inh = np.array(
                [post for post in self._external_connections[
                    self._external_connections["receptor_type"]
                    == "inhibitory"]["post"]
                 if post.toNeuronRowOnDLS().toHemisphereOnDLS() == hemisphere])

            _, counts_exc = np.unique(post_exc, return_counts=True)
            _, counts_inh = np.unique(post_inh, return_counts=True)

            counts_max = 0
            for counts in [counts_exc, counts_inh]:
                if len(counts) > 0:
                    counts_max += np.max(counts)

            if counts_max > free_rows[int(hemisphere.toEnum())]:
                raise RuntimeError("""Not enough synapse driver available for
                                   specified external input.""")

        # update the compare mask of the already allocated synapse driver in a
        # way that they only receive input from on-chip events
        for coord, config in self._synapse_drivers:
            padibus = halco.PADIBusOnDLS(
                coord.toPADIBusOnPADIBusBlock(),
                halco.PADIBusBlockOnDLS(
                    coord.toSynapseDriverBlockOnDLS().toEnum()))
            place, _ = np.where(free_padibusses == padibus)
            assert len(place) == 1
            place = place[0]
            config.row_address_compare_mask = free_padibusses[place][1]

    # pylint: disable=too-many-locals
    def _external_configuration(self,
                                connections: np.ndarray,
                                free_rows: np.ndarray,
                                used_entries: np.ndarray,
                                hemisphere: int) -> None:
        """
        Configure the synapse drivers and synapse matrix to forward the
        specified external events.
        """

        # get the already existing synapse matrix config or a default one, if
        # there is not one yet
        synmtx_coord = halco.SynramOnDLS(hemisphere)
        synmtx_config, coord_index = \
            self._find_used_synapse_matrix(synmtx_coord)

        # add connections
        for pre, conns in connections:
            # values for new entry
            bus, row = free_rows[len(used_entries)]
            using_entry = np.array([(bus, row, np.array([]), 0)],
                                   dtype=used_entries.dtype)
            update_entry = False

            # check if an used row still has all entries available for input
            # from pre
            for used_entry in used_entries:
                usable = True
                for post in conns["post"]:
                    if post in used_entry["posts"]:
                        usable = False
                        break
                # if yes, this row will be used
                if usable:
                    bus = used_entry["bus"]
                    row = used_entry["row"]
                    using_entry = used_entry
                    update_entry = True
                    break

            # configure synapse matrix
            rows_per_syndrv = 2
            pre = int((row - row % rows_per_syndrv)
                      * halco.PADIBusOnPADIBusBlock.size
                      + (row % rows_per_syndrv)
                      + int(bus.toEnum()) * rows_per_syndrv)
            posts = []
            addr = using_entry["number_pres"]
            if isinstance(addr, np.ndarray):
                addr = addr[0]
            for conn in conns:
                post_ind = int(conn["post"].toEnum())
                synmtx_config.labels[pre][post_ind] = int(addr)
                synmtx_config.weights[pre][post_ind] = int(conn["weight"])
                posts.append(conn["post"])

            # update self._external_events
            syndrv = halco.SynapseDriverOnSynapseDriverBlock(
                int(halco.SynapseDriverOnPADIBus(row // 2))
                * halco.PADIBusOnPADIBusBlock.size
                + int(bus))
            label = int(bus.toEnum()) << 14 | int(syndrv.toEnum()) << 6 | addr
            external_event = np.array(
                [(conns["spiketimes"][0], label)],
                dtype=self._external_events.dtype)
            self._external_events = np.append(self._external_events,
                                              external_event)

            # update self._synapse_drivers
            receptor_type = conns["receptor_type"][0]
            global_syndrv = halco.SynapseDriverOnDLS(
                syndrv, halco.SynapseDriverBlockOnDLS(hemisphere))
            used_syndrv = np.array([(global_syndrv, row % 2, pre, posts,
                                     receptor_type)],
                                   dtype=self._used_syndrv.dtype)
            self._used_syndrv = np.append(self._used_syndrv, used_syndrv)
            mask = 0b11111
            self._configure_synapse_drivers(used_syndrv, mask)

            # update used_entry
            try:
                using_entry["posts"] = np.append(using_entry["posts"], posts)
            except ValueError:
                using_entry["posts"][0] = np.append(using_entry["posts"][0],
                                                    posts)
            using_entry["number_pres"] += 1
            if not update_entry:
                used_entries = np.append(used_entries, using_entry)

        # update self._synapse_matrices
        if len(used_entries) > 0:
            self._update_synapse_matrix(coord_index,
                                        synmtx_coord,
                                        synmtx_config)

        return used_entries


class _State(BaseState):
    """Represent the simulator state."""

    max_weight: ClassVar[int] = halco.SynapseRowOnSynram.size \
        * lola.SynapseMatrix.Weight.max

    # pylint: disable=invalid-name
    def __init__(self):
        super(_State, self).__init__()

        self.spikes = []
        self.times = []
        self.membrane = []

        self.mpi_rank = 0        # disabled
        self.num_processes = 1   # number of MPI processes
        self.running = False
        self.t = 0
        self.t_start = 0
        # TODO: replace by calculation (cf. feature #3594)
        self.dt = 3.4e-05        # average time between two MADC samples
        self.min_delay = 0
        self.max_delay = 0
        self.neuron_placement = []
        self.populations = []
        self.recorders = set([])
        self.connections = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.enable_neuron_bypass = False
        self.log = logger.get("pyNN.brainscales2")
        self.checked_hardware = False

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.connections = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.enable_neuron_bypass = False
        self.checked_hardware = False

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    @staticmethod
    def check_neuron_placement_lut(id_list: list) -> list:
        # TODO: support multi chip
        cell_id_size = halco.AtomicNeuronOnDLS.size
        if len(id_list) > cell_id_size:
            raise ValueError("Too many elements in HW LUT.")
        if len(id_list) > len(set(id_list)):
            raise ValueError("Non unique entries in HW LUT.")
        for index in id_list:
            if not 0 <= index < cell_id_size:
                raise ValueError(
                    "NeuronPermutation list entry out of range."
                    + f"0 < {index} < {cell_id_size}"
                )
        return id_list

    @staticmethod
    def get_spikes(spikes: np.ndarray, runtime: int) -> np.ndarray:
        spiketime = spikes["chip_time"] \
            / (int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        label = spikes["label"]
        label = label[(spiketime <= runtime)]
        spiketime = spiketime[spiketime <= runtime]

        # calculate neuron label
        hemisphere = (label & (0b1 << 5)) << 3
        anncore = (label & (0b1 << 10)) >> 3
        spl1 = (label & (0b11 << 14)) >> 9
        addr = label & 31
        label = hemisphere | anncore | spl1 | addr

        return_spikes = np.array((label, spiketime)).T
        return return_spikes

    @staticmethod
    def get_v(v_input: np.ndarray) -> [np.ndarray, np.ndarray]:
        membrane = v_input["value"][1:]
        times = v_input["chip_time"][1:] \
            / (int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        return times, membrane

    @staticmethod
    def configure_common(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:

        # set global cells
        neuron_params = {
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_exc: 1008,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_inh: 1008,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_threshold_comparator:
            500}

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for key, value in neuron_params.items():
                builder.write(halco.CapMemCellOnDLS(key, block),
                              hal.CapMemCell(value))

        # disables event recording during configuration time
        rec_config = hal.EventRecordingConfig()
        rec_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        # set all neurons on chip to default values
        default_neuron = HXNeuron.lola_from_dict({})
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            builder.write(coord, default_neuron)

        config = hal.CommonNeuronBackendConfig()
        config.clock_scale_fast = 3
        config.clock_scale_slow = 3
        config.enable_clocks = True
        config.enable_event_registers = True
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)
        return builder

    # pylint: disable=unsupported-assignment-operation
    @staticmethod
    def configure_madc(builder: sta.PlaybackProgramBuilder,
                       coord: halco.AtomicNeuronOnDLS) \
            -> sta.PlaybackProgramBuilder:
        """Connect MADC to neuron on given coord."""

        hemisphere = coord.toNeuronRowOnDLS().toHemisphereOnDLS()
        is_odd = (int(coord.toNeuronColumnOnDLS()) % 2) == 1
        is_even = not is_odd

        config = hal.ReadoutSourceSelection()
        smux = config.SourceMultiplexer()
        smux.neuron_odd[halco.HemisphereOnDLS(hemisphere)] = is_odd
        smux.neuron_even[halco.HemisphereOnDLS(hemisphere)] = is_even
        config.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0), smux)
        config.enable_buffer_to_pad[
            halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        # TODO: move to Recorder class (cf. feature #3595)
        # set capmem cells
        readout_params = {
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_0: 0,
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_1: 0,
            halco.CapMemCellOnDLS.readout_pseudo_diff_buffer_bias: 0,
            halco.CapMemCellOnDLS.readout_ac_mux_i_bias: 500,
            halco.CapMemCellOnDLS.readout_madc_in_500na: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_i_bias: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_v_ref: 400,
            halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref: 400,
            halco.CapMemCellOnDLS.readout_iconv_test_voltage: 400,
            halco.CapMemCellOnDLS.readout_iconv_sc_amp_v_ref: 400,
        }

        for key, variable in readout_params.items():
            builder.write(key, hal.CapMemCell(variable))

        return builder

    @staticmethod
    def configure_hxneurons(builder: sta.PlaybackProgramBuilder,
                            population: Population,
                            enable_spike_recording: bool,
                            enable_v_recording: bool) \
            -> sta.PlaybackProgramBuilder:
        """
        Places Neurons in Population "pop" on chip and configures spike and
        v recording.
        """

        # configures a neuron with the passed initial values
        atomic_neuron = HXNeuron.lola_from_dict(population.initial_values)

        # places the neurons from pop on chip
        # TODO: derive coord from something else than all_cells
        #       (cf. feature #3687)
        for coord in population.all_cells:
            coord = state.neuron_placement[coord]
            coord = halco.AtomicNeuronOnDLS(
                halco.AtomicNeuronOnDLS.enum_type(coord))

            # configure spike recording
            if enable_spike_recording:
                # neurons per crossbar input channel
                neurons = int(halco.NeuronColumnOnDLS.size
                              / halco.NeuronEventOutputOnDLS.size)
                # arbitrary shift to leave 0 open
                offset = 64
                addr = (coord.toNeuronColumnOnDLS().toEnum() % neurons) \
                    + (coord.toNeuronRowOnDLS().toEnum() * neurons) + offset

                atomic_neuron.event_routing.enable_analog = True
                atomic_neuron.event_routing.enable_digital = True
                atomic_neuron.event_routing.address = int(addr)
                if state.enable_neuron_bypass:
                    # disable threshold comparator
                    atomic_neuron.threshold.enable = False
                    atomic_neuron.event_routing.enable_bypass_excitatory = True
                    atomic_neuron.event_routing.enable_bypass_inhibitory = True
                atomic_neuron.leak_reset.i_bias_source_follower = 280

            # configure v recording
            if enable_v_recording:
                atomic_neuron.event_routing.enable_analog = True
                atomic_neuron.event_routing.enable_digital = True
                atomic_neuron.leak_reset.i_bias_source_follower = 280
                atomic_neuron.readout.enable_amplifier = True
                atomic_neuron.readout.enable_buffered_access = True
                atomic_neuron.readout.i_bias = 1000
                builder = state.configure_madc(builder, coord)

            builder.write(coord, atomic_neuron)

        return builder

    @staticmethod
    def configure_recorders_populations(builder: sta.PlaybackProgramBuilder) \
            -> (sta.PlaybackProgramBuilder, bool):

        v_recording = False
        for recorder in state.recorders:
            population = recorder.population
            assert type(population.celltype) in [HXNeuron, SpikeSourceArray]
            if isinstance(population.celltype, HXNeuron):
                enable_spike_recording = False
                enable_v_recording = False
                for variable in recorder.recorded:
                    if variable == "spikes":
                        enable_spike_recording = True
                    elif variable == "v":
                        enable_v_recording = True
                        v_recording = True
                    else:
                        raise NotImplementedError
                builder = state.configure_hxneurons(
                    builder,
                    population,
                    enable_spike_recording=enable_spike_recording,
                    enable_v_recording=enable_v_recording)

        return builder, v_recording

    @staticmethod
    def madc_configuration(builder: sta.PlaybackProgramBuilder,
                           runtime: float) \
            -> sta.PlaybackProgramBuilder:
        """Configure number of MADC samples depending on set runtime."""
        config = hal.MADCConfig()
        number_of_samples = int(runtime / state.dt)
        if number_of_samples > hal.MADCConfig.NumberOfSamples.max:
            raise ValueError(
                "Recording time is limited to "
                + f"{hal.MADCConfig.NumberOfSamples.max * state.dt}ms"
                + f" ({hal.MADCConfig.NumberOfSamples.max} samples)."
            )
        config.number_of_samples = number_of_samples
        builder.write(halco.MADCConfigOnDLS(), config)
        return builder

    @staticmethod
    def madc_arm_recording(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """Arm MADC recording: Enable wakeup, needs some settle time
        afterwards)."""
        config = hal.MADCControl()
        config.enable_power_down_after_sampling = True
        config.start_recording = False
        config.wake_up = True
        config.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), config)
        return builder

    @staticmethod
    def madc_start_recording(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """Start MADC recording."""
        config = hal.MADCControl()
        config.enable_power_down_after_sampling = True
        config.start_recording = True
        config.wake_up = False
        config.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), config)
        return builder

    @staticmethod
    def configure_crossbar(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:

        # The used numbers can easily be understood looking at a figure of the
        # crossbar (cf. https://github.com/electronicvisions/halco/blob/master
        # /include/halco/hicann-dls/vx/routing_crossbar.h#L77)

        active_crossbar_node = hal.CrossbarNode()
        silent_crossbar_node = hal.CrossbarNode().drop_all

        # enable recurrent connections within top half
        for cinput in range(8):
            builder.write(
                halco.CrossbarNodeOnDLS(halco.CrossbarOutputOnDLS(cinput % 4),
                                        halco.CrossbarInputOnDLS(cinput)),
                active_crossbar_node)

        # enable L2 output
        for cinput in range(8):
            builder.write(
                halco.CrossbarNodeOnDLS(
                    halco.CrossbarOutputOnDLS(8 + cinput % 4),
                    halco.CrossbarInputOnDLS(cinput)),
                active_crossbar_node)

        # clear all inputs
        for coutput in range(8):
            for cinput in range(8, 12):
                builder.write(
                    halco.CrossbarNodeOnDLS(halco.CrossbarOutputOnDLS(coutput),
                                            halco.CrossbarInputOnDLS(cinput)),
                    silent_crossbar_node)

        # disable loopback
        for cinput in range(4):
            builder.write(
                halco.CrossbarNodeOnDLS(halco.CrossbarOutputOnDLS(8 + cinput),
                                        halco.CrossbarInputOnDLS(8 + cinput)),
                silent_crossbar_node)

        # enable input from L2 to top half
        for coutput in range(4):
            builder.write(
                halco.CrossbarNodeOnDLS(halco.CrossbarOutputOnDLS(coutput),
                                        halco.CrossbarInputOnDLS(8 + coutput)),
                active_crossbar_node)

        # enable input from L2 to bottom half
        for coutput in range(4, 8):
            builder.write(
                halco.CrossbarNodeOnDLS(halco.CrossbarOutputOnDLS(coutput),
                                        halco.CrossbarInputOnDLS(4 + coutput)),
                active_crossbar_node)

        # TODO: Incorporate background spike sources (cf. feature #3648)

        return builder

    @staticmethod
    def configure_routing(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """Does some general configurations if projections are used."""

        # set all synapse drivers to default
        synapse_driver_default = hal.SynapseDriverConfig()
        for syndrv in halco.iter_all(halco.SynapseDriverOnDLS):
            builder.write(syndrv, synapse_driver_default)

        # configure PADI bus
        padi_config = hal.CommonPADIBusConfig()
        for block in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_config.enable_spl1[block] = True
            if state.enable_neuron_bypass:
                # magic number (suggested by JWW)
                padi_config.dacen_pulse_extension[block] = 4
        for padibus in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            builder.write(padibus, padi_config)

        # configure switches
        current_switch_quad = hal.ColumnCurrentQuad()
        switch = current_switch_quad.ColumnCurrentSwitch()
        switch.enable_synaptic_current_excitatory = True
        switch.enable_synaptic_current_inhibitory = True

        for s in halco.iter_all(halco.EntryOnQuad):
            current_switch_quad.set_switch(s, switch)

        for sq in halco.iter_all(halco.ColumnCurrentQuadOnDLS):
            builder.write(sq, current_switch_quad)

        builder = state.configure_crossbar(builder)

        # set synapse capmem cells
        synapse_params = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 300,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 1010,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 1010,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 1010}
        if state.enable_neuron_bypass:
            synapse_params[halco.CapMemCellOnCapMemBlock.syn_i_bias_dac] = 1022

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            for k, v in synapse_params.items():
                builder.write(halco.CapMemCellOnDLS(k, block),
                              hal.CapMemCell(v))

        return builder

    @staticmethod
    def run_on_chip(builder: sta.PlaybackProgramBuilder, runtime: float,
                    v_recording: bool, events: np.ndarray) \
            -> sta.PlaybackProgramBuilder:
        """
        Runs the experiment on chip and records MADC samples, if there is a
        recorder recording 'v'.
        """

        # enable event recording
        rec_config = hal.EventRecordingConfig()
        rec_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        if v_recording:
            builder = state.madc_arm_recording(builder)

        # wait 100 us to buffer some program in FPGA to reach precise timing
        # afterwards and sync time
        initial_wait = 100  # us
        builder.write(halco.TimerOnDLS(), hal.Timer())
        builder.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())

        if v_recording:
            builder = state.madc_start_recording(builder)

        # insert events
        for time, label in events:
            builder.block_until(
                halco.TimerOnDLS(),
                int((initial_wait + time * 1e3)
                    * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
            builder.write(
                halco.SpikePack1ToChipOnDLS(),
                hal.SpikePack1ToChip([
                    hal.SpikeLabel(int(label))]))

        # record for time 'runtime'
        builder.block_until(halco.TimerOnDLS(), hal.Timer.Value(
            int(int(hal.Timer.Value.fpga_clock_cycles_per_us)
                * ((runtime * 1000) + initial_wait))))

        # disable event recording
        rec_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        return builder

    def _check_link_notifications(self, link_notifications):
        """
        Check for unexpected link notifications and log accordingly.
        """
        notis_per_phy = dict()
        for noti in link_notifications:
            if noti.link_up and noti.phy not in notis_per_phy.keys():
                # first link up message per phy is expected
                pass
            else:
                # everything else is not expected
                self.log.WARN(noti)
            notis_per_phy[noti.phy] = noti

        if all(not noti.link_up for noti in notis_per_phy.values()):
            self.log.ERROR("All highspeed links down at "
                           + "the end of the experiment.")

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

        sta.run(connection, builder.done())

        error_msg = "_perform_post_fail_analysis(): "
        error_msg += "Experiment failed, reading post-mortem status."
        error_msg += str(ticket_arq.get())
        for ticket_phy in tickets_phy:
            error_msg += str(ticket_phy.get()) + "\n"
        self.log.ERROR(error_msg)

    @staticmethod
    def _perform_hardware_check(connection):
        """
        Check hardware for requirements such as chip version.
        """
        # perform chip-version check
        builder_chip_version, _ = sta.DigitalInit().generate()
        jtag_id_ticket = builder_chip_version.read(halco.JTAGIdCodeOnDLS())
        sta.run(connection, builder_chip_version.done())
        chip_version = jtag_id_ticket.get().version.value()
        if chip_version != 0:
            raise RuntimeError("Unexpected chip version: "
                               + str(chip_version))

    def run(self, runtime):
        self.t += runtime
        self.running = True

        # initialize chip
        builder1, _ = sta.ExperimentInit().generate()

        # common configuration
        builder1 = self.configure_common(builder1)

        # configure populations and recorders
        builder1, v_recording = self.configure_recorders_populations(builder1)

        if v_recording:
            builder1 = self.madc_configuration(builder1, runtime)

        external_events = []
        if len(self.connections) != 0:
            builder1 = self.configure_routing(builder1)
            connection_builder = ConnectionConfigurationBuilder()
            for connection in self.connections:
                # distribute high weights over multiple synapse rows
                if connection.weight > lola.SynapseMatrix.Weight.max:
                    conns_full = connection.weight \
                        // lola.SynapseMatrix.Weight.max
                    weight_left = connection.weight \
                        % lola.SynapseMatrix.Weight.max
                    connection.weight = lola.SynapseMatrix.Weight.max
                    for _ in range(int(conns_full)):
                        connection_builder.add(connection)
                    if weight_left > 0:
                        connection.weight = weight_left
                        connection_builder.add(connection)
                else:
                    connection_builder.add(connection)
            connection_builder_return, external_events = \
                connection_builder.generate()
            builder1.merge_back(connection_builder_return)

        # wait 20000 us for capmem voltages to stabilize
        initial_wait = 20000  # us
        builder1.write(halco.TimerOnDLS(), hal.Timer())
        builder1.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

        builder2 = sta.PlaybackProgramBuilder()
        builder2 = self.run_on_chip(builder2, runtime, v_recording,
                                    external_events)

        program = builder2.done()

        with hxcomm.ManagedConnection() as conn:
            if not self.checked_hardware:
                self._perform_hardware_check(conn)
                self.checked_hardware = True
            try:
                sta.run(conn, builder1.done())
                sta.run(conn, program)
            except RuntimeError:
                # report hs link notifications in any case
                self._check_link_notifications(
                    program.highspeed_link_notifications)
                # perform post-mortem read out of status
                self._perform_post_fail_analysis(conn)
                raise

        # make list 'spikes' of tupel (neuron id, spike time)
        self.spikes = self.get_spikes(program.spikes.to_numpy(), runtime)

        # make two list for madc samples: times, membrane
        self.times, self.membrane = self.get_v(program.madc_samples.to_numpy())

        # warn if unexpected highspeed link notifications have been received
        self._check_link_notifications(program.highspeed_link_notifications)


state = _State()  # a Singleton, so only a single instance ever exists
