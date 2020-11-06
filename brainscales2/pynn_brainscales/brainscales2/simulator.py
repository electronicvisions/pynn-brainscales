# pylint: disable=too-many-lines

from typing import Optional, Set, Tuple, Final, List, Dict, Union
import numpy as np
from pyNN.common import IDMixin, Connection
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells import HXNeuron, \
    SpikeSourceArray
from dlens_vx_v2 import hal, halco, sta, hxcomm, lola, logger


name = "HX"  # for use in annotating output data


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        int.__init__(n)
        IDMixin.__init__(self)


class NeuronPlacement:
    # TODO: support multi compartment issue #3750
    """
    Tracks assignment of pyNN IDs of HXNeuron based populations to the
    corresponding hardware entity, i.e. AtomicNeuronOnDLS. Default constructed
    with 1 to 1 permutation.

    :param neuron_id: Look up table for permutation. Index: HW related
                    population neuron enumeration. Value: HW neuron
                    enumeration.
    """
    _id_2_an: Dict[ID, halco.AtomicNeuronOnDLS]
    _permutation: List[halco.AtomicNeuronOnDLS]
    _max_num_entries: Final[int] = halco.AtomicNeuronOnDLS.size
    default_permutation: Final[List[int]] = range(halco.AtomicNeuronOnDLS.size)

    def __init__(self, permutation: List[int] = None):
        if permutation is None:
            permutation = range(self._max_num_entries)
        self._id_2_an = dict()
        self._permutation = self._check_and_transform(permutation)

    def register_id(self, neuron_id: Union[List[ID], ID]):
        """
        Register a new ID to placement

        :param neuron_id: pyNN neuron ID to be registered
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if len(self._id_2_an) + len(neuron_id) > len(self._permutation):
            raise ValueError(
                f"Cannot register more than {len(self._permutation)} IDs")
        for idx in neuron_id:
            self._id_2_an[idx] = self._permutation[len(self._id_2_an)]

    def id2atomicneuron(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[halco.AtomicNeuronOnDLS], halco.AtomicNeuronOnDLS]:
        """
        Get hardware coordinate from pyNN ID

        :param neuron_id: pyNN neuron ID
        """
        try:
            return [self._id_2_an[idx] for idx in neuron_id]
        except TypeError:
            return self._id_2_an[neuron_id]

    def id2hwenum(self, neuron_id: Union[List[ID], ID]) \
            -> Union[List[int], int]:
        """
        Get hardware coordinate as plain int from pyNN ID

        :param neuron_id: pyNN neuron ID
        """
        atomic_neuron = self.id2atomicneuron(neuron_id)
        try:
            return [int(idx.toEnum()) for idx in atomic_neuron]
        except TypeError:
            return int(atomic_neuron.toEnum())

    @staticmethod
    def _check_and_transform(lut: list) -> list:

        cell_id_size = NeuronPlacement._max_num_entries
        if len(lut) > cell_id_size:
            raise ValueError("Too many elements in HW LUT.")
        if len(lut) > len(set(lut)):
            raise ValueError("Non unique entries in HW LUT.")
        permutation = []
        for neuron_idx in lut:
            if not 0 <= neuron_idx < cell_id_size:
                raise ValueError(
                    f"NeuronPermutation list entry {neuron_idx} out of range. "
                    + f"Needs to be in range [0, {cell_id_size - 1}]"
                )
            coord = halco.AtomicNeuronOnDLS(halco.common.Enum(neuron_idx))
            permutation.append(coord)
        return permutation


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

    def generate(self) -> Tuple[sta.PlaybackProgramBuilder, np.ndarray]:
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
            post = state.neuron_placement.\
                id2atomicneuron(connection.postsynaptic_cell)
            weight = connection.weight
            receptor_type = connection.projection.receptor_type
            external_connection = np.array(
                [(pre, post, weight, receptor_type, pre.spike_times.value)],
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

        pre = state.neuron_placement.id2hwenum(connection.presynaptic_cell)
        post = state.neuron_placement.id2hwenum(connection.postsynaptic_cell)

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
            self._used_padi_rows[int(padibus_index)][1] = padi_row_index

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

        nrn = state.neuron_placement. \
            id2atomicneuron(connection.postsynaptic_cell)
        return nrn.toNeuronRowOnDLS().toSynramOnDLS()

    def _find_used_synapse_matrix(self, synmtx_coord: halco.SynramOnDLS) \
            -> Tuple[lola.SynapseMatrix, np.ndarray]:
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

        pre = state.neuron_placement. \
            id2atomicneuron(connection.presynaptic_cell)
        post = state.neuron_placement.id2hwenum(connection.postsynaptic_cell)
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
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def _external_configuration(self,
                                connections: np.ndarray,
                                free_rows: np.ndarray,
                                used_entries: np.ndarray,
                                hemisphere: int) -> np.ndarray:
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

            # Synapse driver used for the current connection
            using_driver = halco.SynapseRowOnSynram(row). \
                toSynapseDriverOnSynapseDriverBlock()

            # Check what synapse labels are in use for the current synapse
            # driver, use the following one.
            # It is enough to keep labels unique per synapse driver, because
            # row address select mask 0b11111 is used for external inputs.
            addr: int = hal.SynapseLabelValue.min
            used_labels: Set[int] = set()
            for entry in used_entries:
                used_driver = halco.SynapseRowOnSynram(entry["row"]). \
                    toSynapseDriverOnSynapseDriverBlock()
                if using_driver == used_driver:
                    used_labels.add(entry["number_pres"])
            if len(used_labels) > 0:
                addr = max(used_labels) + 1
            if addr > hal.SynapseLabelValue.max:
                raise RuntimeError("Label too large, cannot map the network.")
            using_entry["number_pres"] = addr

            # configure synapse matrix
            rows_per_syndrv = 2
            pre = int((row - row % rows_per_syndrv)
                      * halco.PADIBusOnPADIBusBlock.size
                      + (row % rows_per_syndrv)
                      + int(bus.toEnum()) * rows_per_syndrv)
            posts = []
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
            label = int(bus.toEnum()) << 14 | \
                int(syndrv.toSynapseDriverOnPADIBus().toEnum()) << 6 | \
                addr
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
            if not update_entry:
                used_entries = np.append(used_entries, using_entry)

        # update self._synapse_matrices
        if len(used_entries) > 0:
            self._update_synapse_matrix(coord_index,
                                        synmtx_coord,
                                        synmtx_config)

        return used_entries


class State(BaseState):
    """Represent the simulator state."""

    max_weight: Final[int] = halco.SynapseRowOnSynram.size \
        * lola.SynapseMatrix.Weight.max

    # pylint: disable=invalid-name
    # TODO: replace by calculation (cf. feature #3594)
    dt: Final[float] = 3.4e-05  # average time between two MADC samples

    # pylint: disable=invalid-name
    def __init__(self):
        super(State, self).__init__()

        self.spikes = []
        self.times = []
        self.madc_samples = []

        self.mpi_rank = 0        # disabled
        self.num_processes = 1   # number of MPI processes
        self.running = False
        self.t = 0
        self.t_start = 0
        self.min_delay = 0
        self.max_delay = 0
        self.neuron_placement = None
        self.populations = []
        self.recorders = set([])
        self.madc_recorder = None
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
        self.populations = []
        self.madc_recorder = None
        self.connections = []
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.enable_neuron_bypass = False
        self.checked_hardware = False
        self.neuron_placement = None

        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

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
    def get_v(v_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        madc_samples = v_input["value"][1:]
        times = v_input["chip_time"][1:] \
            / (int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)

        # samples may be returned out of order -> sort here
        times_sorted_indices = times.argsort()
        times = times[times_sorted_indices]
        madc_samples = madc_samples[times_sorted_indices]

        return times, madc_samples

    @staticmethod
    def configure_common(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:

        # set global cells
        neuron_params = {
            halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n: 340,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp: 110,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_leak_source_follower:
            100,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_spike_comparator:
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
        default_neuron = HXNeuron.create_hw_entity({})
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

    # pylint: disable=too-many-arguments
    def configure_hxneuron(self,
                           builder: sta.PlaybackProgramBuilder,
                           neuron_id: ID,
                           parameters: dict,
                           enable_spike_recording: bool,
                           readout_source:
                           Optional[hal.NeuronConfig.ReadoutSource]) \
            -> sta.PlaybackProgramBuilder:
        """
        Places Neuron in Population "pop" on chip and configures spike and
        v recording.

        :param enable_spike_recording: toggles if spikes are send off chip;
                                       ignored at the moment
        """

        assert enable_spike_recording

        # places the neurons from pop on chip
        atomic_neuron = HXNeuron.create_hw_entity(parameters)
        coord = self.neuron_placement.id2atomicneuron(neuron_id)

        # configure spike recording
        # neurons per crossbar input channel
        neurons = int(halco.NeuronColumnOnDLS.size
                      / halco.NeuronEventOutputOnDLS.size)
        # arbitrary shift to leave 0 open
        offset = 64
        addr = (coord.toNeuronColumnOnDLS().toEnum() % neurons) \
            + (coord.toNeuronRowOnDLS().toEnum() * neurons) + offset

        atomic_neuron.event_routing.analog_output = \
            atomic_neuron.EventRouting.AnalogOutputMode.normal
        atomic_neuron.event_routing.enable_digital = True
        atomic_neuron.event_routing.address = int(addr)
        if self.enable_neuron_bypass:
            # disable threshold comparator
            atomic_neuron.threshold.enable = False
            atomic_neuron.event_routing.enable_bypass_excitatory = True
            atomic_neuron.event_routing.enable_bypass_inhibitory = True

        # configure v recording
        if readout_source is not None:
            atomic_neuron.event_routing.analog_output = \
                atomic_neuron.EventRouting.AnalogOutputMode.normal
            atomic_neuron.event_routing.enable_digital = True
            atomic_neuron.readout.enable_amplifier = True
            atomic_neuron.readout.enable_buffered_access = True
            atomic_neuron.readout.source = readout_source
            builder = self.configure_madc(builder, coord)

        builder.write(coord, atomic_neuron)

        return builder

    def configure_recorders_populations(self,
                                        builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:

        for recorder in self.recorders:
            population = recorder.population
            assert isinstance(population.celltype, (HXNeuron,
                                                    SpikeSourceArray))
            if isinstance(population.celltype, HXNeuron):

                # retrieve for which neurons what kind of recording is active
                spike_rec_indexes = []
                madc_recording_id = None
                readout_source = Optional[hal.NeuronConfig.ReadoutSource]
                for parameter, cell_ids in recorder.recorded.items():
                    for cell_id in cell_ids:
                        # we always record spikes at the moment
                        spike_rec_indexes.append(cell_id)
                        if parameter == "spikes":
                            pass
                        elif parameter in recorder.madc_variables:
                            assert self.madc_recorder is not None and \
                                cell_id == self.madc_recorder.cell_id
                            madc_recording_id = cell_id
                            readout_source = self.madc_recorder.readout_source
                        else:
                            raise NotImplementedError
                for cell_id, parameters in zip(
                        population.all_cells,
                        population.celltype.parameter_space):

                    # we always record spikes at the moment
                    enable_spike_recording = True
                    this_source = None
                    if cell_id == madc_recording_id:
                        this_source = readout_source
                    builder = self.configure_hxneuron(
                        builder,
                        cell_id,
                        parameters,
                        enable_spike_recording=enable_spike_recording,
                        readout_source=this_source)

        return builder

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
                # extend pulse length such that pre-synaptic signals have
                # a stronger effect on the synaptic input voltage and spikes
                # are more easily detected by the bypass circuit.
                padi_config.dacen_pulse_extension[block] = \
                    hal.CommonPADIBusConfig.DacenPulseExtension.max
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
            halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1022,
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
                    have_madc_recording: bool, events: np.ndarray) \
            -> sta.PlaybackProgramBuilder:
        """
        Runs the experiment on chip and records MADC samples, if there is a
        recorder recording some MADC observable.
        """

        # enable event recording
        rec_config = hal.EventRecordingConfig()
        rec_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        if have_madc_recording:
            builder = state.madc_arm_recording(builder)

        # wait 100 us to buffer some program in FPGA to reach precise timing
        # afterwards and sync time
        initial_wait = 100  # us
        builder.write(halco.TimerOnDLS(), hal.Timer())
        builder.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())

        if have_madc_recording:
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

    def _check_link_notifications(self, link_notifications,
                                  n_expected_notifications):
        """
        Check for unexpected link notifications and log accordingly.
        When turning the link on, a link up message is expected.

        :param link_notifications: List of link notifications
        :param n_expected_notifications: Number of expected link up messages
        """
        notis_per_phy = dict()
        for noti in link_notifications:
            if n_expected_notifications > 0 and noti.link_up and \
                    noti.phy not in notis_per_phy.keys():
                # one link up message per phy is expected (when turned on)
                pass
            else:
                # everything else is not expected
                self.log.WARN(noti)
            notis_per_phy[noti.phy] = noti

        if len(notis_per_phy) < n_expected_notifications:
            self.log.ERROR("Not all configured highspeed links sent link "
                           + "notifications.")

        if len(notis_per_phy) == halco.PhyStatusOnFPGA.size and \
                all(not noti.link_up for noti in notis_per_phy.values()):
            self.log.ERROR("All configured highspeed links down at "
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
        if chip_version != 2:
            raise RuntimeError("Unexpected chip version: "
                               + str(chip_version))

    # pylint: disable=too-many-locals,too-many-statements
    def run(self, runtime: Optional[float]):
        """
        Performs a hardware run for `runtime` milliseconds.
        If runtime is `None`, we only perform preparatory steps.
        """
        if runtime is None:
            self.log.INFO("User requested 'None' runtime: "
                          + "no hardware run performed.")
        else:
            self.t += runtime
        self.running = True

        # initialize chip
        builder1, _ = sta.ExperimentInit().generate()

        # common configuration
        builder1 = self.configure_common(builder1)

        have_madc_recording = self.madc_recorder is not None
        # configure populations and recorders
        builder1 = self.configure_recorders_populations(builder1)

        if have_madc_recording:
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

        if runtime is None:
            return

        # wait 20000 us for capmem voltages to stabilize
        initial_wait = 20000  # us
        builder1.write(halco.TimerOnDLS(), hal.Timer())
        builder1.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

        builder2 = sta.PlaybackProgramBuilder()
        builder2 = self.run_on_chip(builder2, runtime, have_madc_recording,
                                    external_events)

        program1 = builder1.done()
        program2 = builder2.done()
        with hxcomm.ManagedConnection() as conn:
            if not self.checked_hardware:
                self._perform_hardware_check(conn)
                self.checked_hardware = True
            try:
                sta.run(conn, program1)
                sta.run(conn, program2)
            except RuntimeError:
                # Link up messages for all links are expected.
                self._check_link_notifications(
                    program1.highspeed_link_notifications,
                    halco.PhyStatusOnFPGA.size)
                # Since the simulation builder (builder2) does not modify the
                # highspeed links, no notifications are expected.
                self._check_link_notifications(
                    program2.highspeed_link_notifications, 0)
                # perform post-mortem read out of status
                self._perform_post_fail_analysis(conn)
                raise

        # make list 'spikes' of tupel (neuron id, spike time)
        self.spikes = self.get_spikes(program2.spikes.to_numpy(), runtime)

        # make two list for madc samples: times, madc_samples
        self.times, self.madc_samples = self.get_v(
            program2.madc_samples.to_numpy())

        # warn if unexpected highspeed link notifications have been received.
        self._check_link_notifications(program1.highspeed_link_notifications,
                                       halco.PhyStatusOnFPGA.size)
        # Since the simulation builder (builder2) does not modify the highspeed
        # links, no notifications are expected.
        self._check_link_notifications(program2.highspeed_link_notifications,
                                       0)


# state is instantiated in setup()
state: Optional[State] = None
