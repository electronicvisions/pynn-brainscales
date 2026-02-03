from itertools import product, compress
from typing import Sequence, Set, List, Optional, Tuple
from datetime import datetime
from warnings import warn
import numpy as np
import neo
import quantities as pq
import pyNN.recording
import pyNN.errors

from pyNN.recording import gather_blocks, filter_by_variables, \
    remove_duplicate_spiketrains

from pynn_brainscales.brainscales2 import simulator
from pynn_brainscales.brainscales2.recording_data import RecordingSite, \
    RecordingConfig, GrenadeRecId, RecordingType
from dlens_vx_v3 import halco


def normalize_variables_arg(variables):
    """If variables is a single string, encapsulate it in a list."""
    if isinstance(variables, str) and variables != 'all':
        return [variables]
    return variables


class Recorder(pyNN.recording.Recorder):

    _simulator = simulator

    def __init__(self, population, file=None):
        super().__init__(population, file=file)
        self.changed_since_last_run = True

    def _get_recording_sites(self, neuron_ids: List[int],
                             locations: Optional[Sequence[str]] = None
                             ) -> Set[RecordingSite]:
        if locations is None:
            # record first compartment if locations is not defined
            comp_ids = [halco.CompartmentOnLogicalNeuron()]
        else:
            celltype = self.population.celltype
            try:
                comp_ids = celltype.get_compartment_ids(locations)
            except AttributeError as err:
                raise RuntimeError('Can not extract recording locations for '
                                   f'celltype "{celltype.__name__}".') from err

        return {RecordingSite(n_id, c_id) for n_id, c_id in
                product(neuron_ids, comp_ids)}

    def record(self, variables, ids, sampling_interval=None,
               locations=None, *, device="madc"):
        self.changed_since_last_run = True
        # MADC based recording is only possible for one neuron on a chip
        # it is therefore checked for population size one and no multi
        # assignment before the parent record function is called which modifies
        # state
        if sampling_interval:
            raise NotImplementedError("Sampling interval not implemented.")
        variable_list = normalize_variables_arg(variables)

        ids = {id for id in ids if id.local}
        recording_sites = self._get_recording_sites(ids, locations)
        grenade_ids = {self._rec_site_to_grenade_index(rec_site) for rec_site
                       in recording_sites}

        if device == "madc":
            self._simulator.state.recordings[-1].config.add_madc_recording(
                set(variable_list).intersection(
                    RecordingConfig.analog_observable_names),
                grenade_ids)
        elif 'pad' in device:
            pad, buffered = self._device_name_to_pad_config(device)
            self._simulator.state.recordings[-1].config.add_pad_readout(
                set(variable_list).intersection(
                    RecordingConfig.analog_observable_names),
                grenade_ids,
                pad=pad, buffered=buffered)
        else:
            raise ValueError(f'Device "{device}" is not supported')

        self._simulator.state.log.debug(f'Recorder.record(<{len(ids)} cells>)')
        for variable in variable_list:
            if not self.population.can_record(variable):
                raise pyNN.errors.RecordingError(variable,
                                                 self.population.celltype)
            # save all recording sites in a set
            self.recorded[variable] |= recording_sites

    @staticmethod
    def _device_name_to_pad_config(device: str) -> Tuple[str, bool]:
        '''
        Extract pad and buffering mode from device name.

        The device name is of the form 'pad_[0|1][|_unbuffered|_buffered]'.
        Raise ValueError if device name does not fit format. Otherwise,
        extract pad as well as buffering mode from device name.
        :param device: Device to record.
        :returns: Pad, buffering enabled.
        '''
        parts = device.split("_")
        correct_start = parts[0] == 'pad'
        correct_pad_number = len(parts) > 1 and parts[1] in ['0', '1']
        correct_buffering = len(parts) < 3 or parts[2] in ['buffered',
                                                           'unbuffered']
        if len(parts) > 3 or not (correct_start and correct_pad_number
                                  and correct_buffering):
            raise ValueError(f'Device "{device}" is not supported')
        pad = int(parts[1])
        unbuffered = 'unbuffered' in device

        return pad, not unbuffered

    # Overwrite base function since `self.recorded` saves sets of
    # RecordingSite. `filter_ids` filters for cell_ids therefore, we have to
    # implement a different filter function
    def filter_recorded(self, variable, filter_ids):
        if filter_ids is None:
            return self.recorded[variable]
        recording_sites = list(self.recorded[variable])
        cell_ids = np.array([x[0] for x in recording_sites])
        return set(compress(recording_sites, np.isin(cell_ids, filter_ids)))

    def _record(self, variable, new_ids, sampling_interval=None):
        pass

    def _reset(self):
        self.changed_since_last_run = True

        for variable in self.recorded:
            for rec_site in self.recorded[variable]:
                grenade_id = self._rec_site_to_grenade_index(rec_site)
                # this removes the recording site from the configuration as
                # well as previously recorded data since the data can no longer
                # be retrieved.
                self._simulator.state.recordings[-1].remove(
                    recording_site=grenade_id)

    def _clear_simulator(self):
        # here we only remove the recorded data but keep the configuration
        for variable in self.recorded:
            for rec_site in self.recorded[variable]:
                grenade_id = self._rec_site_to_grenade_index(rec_site)
                for recording in self._simulator.state.recordings:
                    recording.data.remove(
                        recording_site=grenade_id)

    def _get_spiketimes(self, ids, snippet_idx, clear=False):
        """Returns a dict containing the recording site and its spiketimes."""
        all_spiketimes = {}
        recording = self._simulator.state.recordings[snippet_idx]
        for rec_site in ids:
            grenade_id = self._rec_site_to_grenade_index(rec_site)
            if grenade_id in recording.data.spikes:
                all_spiketimes[rec_site] = recording.data.spikes[grenade_id]
            if clear:
                recording.data.remove(recording_site=grenade_id,
                                      recording_type=RecordingType.SPIKES)
        return all_spiketimes

    def _rec_site_to_grenade_index(self, rec_site: RecordingSite
                                   ) -> GrenadeRecId:
        index_on_pop = np.where(
            self.population.all_cells == int(rec_site.cell_id))[0]
        assert len(index_on_pop) == 1
        return GrenadeRecId(simulator.state.populations.index(self.population),
                            index_on_pop[0], int(rec_site.comp_id))

    def _get_location_label(self, comp_id: halco.CompartmentOnLogicalNeuron
                            ) -> str:
        try:
            return self.population.celltype.get_label(comp_id)
        except AttributeError:
            return 'unlabeled'

    def add_spike_trains(self,
                         segment: neo.Segment,
                         snippet_idx: int,
                         *,
                         filter_ids=None,
                         clear: bool = True,
                         ) -> None:
        """
        Add the recorded spike trains to the segment.

        :param segment: Segment to which to add the spike trains.
        :param snippet_idx: Snipped for from which to get the
            spike trains.
        :param filter_ids: Ids of cells for which to get the spike
            trains. If None, the spike trains of all cells are retrieved.
        :param clear: Clear recorded data.
        """
        sids = sorted(self.filter_recorded('spikes', filter_ids))
        data = self._get_spiketimes(sids, snippet_idx, clear=clear)

        t_start = sum(self._simulator.state.runtimes[0:snippet_idx]) * pq.ms
        t_stop = t_start + self._simulator.state.runtimes[snippet_idx] * pq.ms

        for cell, spikes in data.items():
            times = pq.Quantity(spikes, pq.ms)
            times += t_start
            if times.size > 0 and times.max() > t_stop:
                warn("Recorded at least one spike after t_stop")
                times = times[times <= t_stop]
            location = self._get_location_label(cell.comp_id)
            spike_train = neo.SpikeTrain(
                times,
                t_start=t_start,
                t_stop=t_stop,
                units='ms',
                source_population=self.population.label,
                source_id=int(cell.cell_id),
                source_location=location,
                source_compartment=int(cell.comp_id),
                source_index=self.population.id_to_index(int(cell.cell_id)))
            segment.spiketrains.append(spike_train)
            for train in segment.spiketrains:
                train.segment = segment

    def add_recording(self,
                      segment: neo.Segment,
                      snippet_idx: int,
                      variable: str,
                      device: str = "madc",
                      *,
                      filter_ids=None,
                      clear: bool = True,
                      ) -> None:
        """
        Add the recorded samples to the segment.

        :param segment: Segment to which add the data.
        :param snippet_idx: Snipped for from which to get the
            samples.
        :param variable: Name of variable for which to get the data.
        :param device: Device for which get the samples. I.e. CADC or
            MADC.
        :param filter_ids: Ids of cells for which to get the data.
            If None, the samples of all cells are retrieved.
        :param clear: Clear recorded data.
        """
        t_start = sum(self._simulator.state.runtimes[0:snippet_idx]) * pq.ms

        ids = sorted(self.filter_recorded(variable, filter_ids))
        signal_array, times_array = self._get_all_signals(
            variable, ids, snippet_idx, device=device, clear=clear)
        times_array += t_start

        # may be empty if none of the recorded cells are on this MPI node
        if signal_array.size == 0:
            return

        units = self.population.find_units(variable)
        assert self.record_times
        assert signal_array.shape == times_array.shape
        signals = [
            neo.IrregularlySampledSignal(
                np.array(times_array[i], dtype=np.float32),
                np.array(signal_array[i], dtype=np.float32),
                units=units,
                time_units=pq.ms,
                name=variable,
                device=device,
                source_ids=[int(cell_id.cell_id)],
                source_locations=[self._get_location_label(cell_id.comp_id)],
                source_population=self.population.label,
                source_compartments=[int(cell_id.comp_id)],
                array_annotations={
                    "channel_index":
                        [self.population.id_to_index(cell_id.cell_id)]}
            )
            for i, cell_id in enumerate(ids)
        ]
        segment.irregularlysampledsignals.extend(signals)
        for signal in signals:
            signal.segment = segment

    # Compared to upstream pynn, we introduce support for multi-compartmental
    # neuron models and different recording devices.
    def _get_current_segment(self, filter_ids=None, variables='all',
                             clear=False):
        segment = neo.Segment(
            name=f"segment{self._simulator.state.segment_counter:03d}",
            description=self.population.describe(),
            rec_datetime=datetime.now())
        variables_to_include = set(self.recorded.keys())
        if variables != 'all':
            variables_to_include = \
                variables_to_include.intersection(set(variables))
        for snippet_idx in range(len(self._simulator.state.recordings) - 1):
            for variable in sorted(variables_to_include):
                if variable == 'spikes':
                    self.add_spike_trains(segment, snippet_idx,
                                          filter_ids=filter_ids, clear=clear)
                else:
                    self.add_recording(segment, snippet_idx,
                                       variable=variable,
                                       device="madc",
                                       filter_ids=filter_ids, clear=clear)
        return segment

    def _get_all_signals(self, variable, ids, snippet_idx,
                         *,
                         clear=False,
                         device: str = "madc"):
        times = []
        values = []
        recording = self._simulator.state.recordings[snippet_idx]
        for cell in ids:
            if cell not in self.recorded.get(variable, set()):
                raise RuntimeError("No samples were recorded for population "
                                   f"'{self.population.label}' at recording "
                                   f"{cell}.")
            grenade_id = self._rec_site_to_grenade_index(cell)
            if grenade_id not in recording.config.analog_observables:
                values.append([])
                times.append([])
                continue

            recorded_var = recording.config.analog_observables[grenade_id]
            if recorded_var != RecordingConfig.str_to_source_map.get(variable):
                raise RuntimeError(f"'{recorded_var}' was recorded but "
                                   f"'{variable}' was requested "
                                   f"(population: {self.population.label}, "
                                   f"recording_site: {id}).")
            if not hasattr(recording.data, device):
                raise RuntimeError(f"No data for device '{device}'.")
            recording_data = getattr(recording.data, device)
            if grenade_id not in recording_data:
                # no samples have been recorded yet
                continue

            data = recording_data[grenade_id]
            times.append(data.times)
            values.append(data.values)
            if clear:
                recording.data.remove(
                    recording_site=grenade_id,
                    recording_type=RecordingType[device.upper()])

        return np.array(values, dtype=object), np.array(times, dtype=object)

    def _local_count(self, variable, filter_ids):
        """
        Count number of spikes for the given filter_ids.
        """
        if variable != "spikes":
            raise ValueError("Only implemented for spikes")
        counts = {}
        for filter_id in filter_ids:
            count = 0
            for recording_site in self.filter_recorded(variable,
                                                       filter_ids):
                for snippet_idx in \
                        range(len(self._simulator.state.recordings) - 1):
                    spikes = self._get_spiketimes(recording_site,
                                                  snippet_idx, clear=False)
                    count += len(spikes)
            counts[int(filter_id)] = len(count)
        return counts

    # TODO: align handling of varibales to PyNN 0.12
    # this function was copied from PyNN 0.10.1 and the agruments were
    # adjusted for PyNN 0.12
    # pylint: disable=too-many-arguments
    def get(self, variables, gather=False, filter_ids=None, clear=False,
            annotations=None, locations=None):
        """Return the recorded data as a Neo `Block`."""
        del locations  # we do not yet support PyNN 0.12
        variables = normalize_variables_arg(variables)
        data = neo.Block()
        data.segments = [filter_by_variables(segment, variables)
                         for segment in self.cache]
        if self._simulator.state.running:
            # reset() has not been called, so current segment is not in cache
            data.segments.append(self._get_current_segment(
                filter_ids=filter_ids, variables=variables, clear=clear))
        for segment in data.segments:
            segment.block = data
        data.name = self.population.label
        data.description = self.population.describe()
        data.rec_datetime = data.segments[0].rec_datetime
        data.annotate(**self.metadata)
        if annotations:
            data.annotate(**annotations)
        if gather and self._simulator.state.num_processes > 1:
            data = gather_blocks(data)
            if (
                hasattr(self.population.celltype, "always_local")
                and self.population.celltype.always_local
            ):
                data = remove_duplicate_spiketrains(data)
        if clear:
            self.clear()
        return data
