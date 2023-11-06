from enum import Enum, auto
from typing import NamedTuple, Set, List, Dict, Optional
import numpy as np

import pygrenade_vx as grenade
from dlens_vx_v3 import hal, halco


class RecordingType(Enum):
    MADC = auto()
    PAD = auto()
    SPIKES = auto()


class RecordingSite(NamedTuple):
    cell_id: int  # Type pynn.simulator.ID (cyclic import)
    comp_id: halco.CompartmentOnLogicalNeuron


class GrenadeRecId(NamedTuple):
    population: int
    neuron_on_population: int
    compartment_on_neuron: halco.CompartmentOnLogicalNeuron


class MADCData(NamedTuple):
    '''
    Times and values of a MADC recording.
    '''
    times: np.ndarray
    values: np.ndarray


class PadConfig(NamedTuple):
    rec_site: GrenadeRecId
    buffered: bool


class Recording:
    """
    Save recording information as well as recorded data.
    """
    def __init__(self) -> None:
        self.config = RecordingConfig()
        self.data = RecordingData()

    def remove(self, recording_site: GrenadeRecId):
        self.config.remove(recording_site)
        self.data.remove(recording_site)


class RecordingData:
    """
    Save recorded data.

    Currently we only support MADC recordings but in future we want to support
    readout via the pads and via the CADC.
    This class saves data which was recorded during an experiment run.
    """
    def __init__(self) -> None:
        self.madc: Dict[GrenadeRecId, MADCData] = {}
        self.spikes: Dict[GrenadeRecId, List[float]] = {}

    def __getitem__(self, key: RecordingType):
        return self.__getattribute__(key.name.lower())

    def __setitem__(self, key: RecordingType, value: Dict):
        self.__setattr__(key.name.lower(), value)

    def remove(self, recording_site: Optional[GrenadeRecId] = None,
               recording_type: Optional[RecordingType] = None):
        recording_types = [RecordingType.MADC, RecordingType.SPIKES]
        if recording_type is not None:
            recording_types = [recording_type]
        for data_type in recording_types:
            if recording_site is None:
                self[data_type] = {}
                continue
            try:
                self[data_type].pop(recording_site)
            except KeyError:
                # samples might already be deleted (for example when
                # `get_data(clear=True)` is called.)
                pass


class RecordingConfig:
    """
    Save which observables are recorded with which "device".

    Currently we only support MADC recordings but in future we want to support
    readout via the pads and via the CADC.
    This class saves which observables are recorded for which recording site
    (recording site = neuron + compartment) and which device is used, where a
    device is the method of readout, i.e. MADC.
    """

    str_to_source_map = \
        {"v": hal.NeuronConfig.ReadoutSource.membrane,
         "exc_synin": hal.NeuronConfig.ReadoutSource.exc_synin,
         "inh_synin": hal.NeuronConfig.ReadoutSource.inh_synin,
         "adaptation": hal.NeuronConfig.ReadoutSource.adaptation}
    analog_observable_names = list(str_to_source_map)

    def __init__(self) -> None:
        self.analog_observables: Dict[GrenadeRecId,
                                      hal.NeuronConfig.ReadoutSource] = {}
        self.madc: List[GrenadeRecId] = []
        self.pads: Dict[halco.PadOnDLS, PadConfig] = {}

    def add_madc_recording(self, variables: Set[str],
                           recording_sites: Set[GrenadeRecId]):
        if len(variables) == 0 or len(recording_sites) == 0:
            return
        if len(variables) > 1:
            raise ValueError("Can only set 1 analog record type per neuron.")
        if len(recording_sites) > 2:
            raise ValueError("Can only record two neurons/locations via MADC.")

        variable = next(iter(variables))
        if variable not in self.analog_observable_names:
            raise RuntimeError(f"Can not record variable '{variable}' with "
                               "the MADC.")
        readout_source = self.str_to_source_map[variable]

        # check if variable already recorded for given sites.
        # Perform check before other loop in order to add all or none sites.
        for recording_site in recording_sites:
            if recording_site in self.analog_observables and \
                    self.analog_observables[recording_site] != readout_source:
                raise ValueError("Only one source can be recorded per neuron.")
        if len(self.madc) + len(recording_sites) > 2:
            raise ValueError("Can only record at most two neurons/locations "
                             "via MADC")

        for recording_site in recording_sites:
            if recording_site in self.madc:
                continue
            self.madc.append(recording_site)
            self.analog_observables[recording_site] = readout_source

    def add_pad_readout(self, variables: Set[str],
                        recording_sites: Set[GrenadeRecId],
                        pad: int,
                        buffered: bool = True):
        if len(variables) == 0 or len(recording_sites) == 0:
            return
        if len(variables) > 1:
            raise ValueError("Can only set 1 analog readout type per neuron.")
        if len(recording_sites) > 1:
            raise ValueError("Can only read out a single neuron per pad.")
        if pad not in [0, 1]:
            raise ValueError("Only pad 0 and 1 are available.")

        variable = next(iter(variables))
        if variable not in self.analog_observable_names:
            raise RuntimeError(f"Can not readout variable '{variable}' at "
                               "the pad.")
        readout_source = self.str_to_source_map[variable]

        # check if other variable is already recorded at given recording sites.
        recording_site = next(iter(recording_sites))
        if recording_site in self.analog_observables and \
                self.analog_observables[recording_site] != readout_source:
            raise ValueError("Only one source can be recorded per neuron.")

        pad_coord = halco.PadOnDLS(pad)
        if pad_coord in self.pads and \
                self.pads[pad_coord] != PadConfig(recording_site, buffered):
            raise ValueError("Can only record one neuron and one mode "
                             "(buffered or unbuffered) per pad.")
        self.pads[pad_coord] = PadConfig(recording_site, buffered)
        self.analog_observables[recording_site] = readout_source

    def remove(self, recording_site: GrenadeRecId):
        if recording_site in self.analog_observables:
            self.analog_observables.pop(recording_site)
        if recording_site in self.madc:
            self.madc.remove(recording_site)

        pads_to_remove = []
        for pad_coord, pad in self.pads.items():
            if pad[0] == recording_site:
                pads_to_remove.append(pad_coord)
        for pad_coord in pads_to_remove:
            del self.pads[pad_coord]

    def add_to_network_graph(self,
                             network_builder: grenade.network.NetworkBuilder
                             ) -> None:
        if len(self.madc) > 0:
            madc_recording_neurons = []
            for rec_site in self.madc:
                source = self.analog_observables[rec_site]
                neuron = grenade.network.MADCRecording.Neuron()
                neuron.coordinate.population = grenade.network\
                    .PopulationOnNetwork(rec_site.population)
                neuron.source = source
                neuron.coordinate.neuron_on_population \
                    = rec_site.neuron_on_population
                neuron.coordinate.compartment_on_neuron \
                    = rec_site.compartment_on_neuron
                neuron.coordinate.atomic_neuron_on_compartment = 0
                madc_recording_neurons.append(neuron)
            madc_recording = grenade.network.MADCRecording(
                madc_recording_neurons)
            network_builder.add(madc_recording)

        if len(self.pads) > 0:
            recordings = {}
            for pad_coord, config in self.pads.items():
                source = self.analog_observables[config.rec_site]

                neuron = grenade.network.MADCRecording.Neuron()
                neuron.coordinate.population = grenade.network\
                    .PopulationOnNetwork(config.rec_site.population)
                neuron.source = source
                neuron.coordinate.neuron_on_population \
                    = config.rec_site.neuron_on_population
                neuron.coordinate.compartment_on_neuron \
                    = config.rec_site.compartment_on_neuron
                neuron.coordinate.atomic_neuron_on_compartment = 0

                recordings[pad_coord] = grenade.network.PadRecording.Source(
                    neuron, config.buffered)
            network_builder.add(grenade.network.PadRecording(recordings))
