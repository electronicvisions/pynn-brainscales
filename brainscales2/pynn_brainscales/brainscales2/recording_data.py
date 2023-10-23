from typing import NamedTuple, Set, List, Dict, Optional
import numpy as np

import pygrenade_vx as grenade
from dlens_vx_v3 import hal, halco


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

    def remove(self, recording_site: Optional[GrenadeRecId] = None):
        for data_type in ['madc', 'spikes']:
            if recording_site is None:
                setattr(self, data_type, {})
            else:
                try:
                    getattr(self, data_type).pop(recording_site)
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

    def remove(self, recording_site: GrenadeRecId):
        if recording_site in self.analog_observables:
            self.analog_observables.pop(recording_site)
        if recording_site in self.madc:
            self.madc.remove(recording_site)

    def add_to_network_graph(self,
                             network_builder: grenade.network.NetworkBuilder
                             ) -> None:
        if len(self.madc) == 0:
            return
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
