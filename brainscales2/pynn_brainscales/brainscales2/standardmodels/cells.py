import inspect
import numbers
from typing import List, Dict, ClassVar, Final, Optional, Union
import numpy as np

from pyNN.parameters import ArrayParameter
from pyNN.standardmodels import cells, build_translations, StandardCellType

from dlens_vx_v2 import lola, hal, halco


class HXNeuron(StandardCellType):
    """
    One to one representation of subset of parameter space of a
    lola.AtomicNeuron. Parameter hierarchy is flattened. Defaults to "silent"
    neuron.

    :param coco_inject: Optional coordinate container pair (coco) injection as
                        a mapping of AtomicNeuronOnDLS to AtomicNeuron.
                        Can be used, e.g. to apply calibration results. If
                        provided default values are replaced with coco entries.
                        Values are applied according to neuron placement.
    :param parameters: Mapping of parameters and corresponding values, e.g.
                       dict. Either 1-dimensional or population size
                       dimensions. Default as well as coco values are
                       overwritten for specified parameters.
    """

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: Final[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                    "adaptation"]
    receptor_types: Final[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: Final[bool] = False
    injectable: Final[bool] = True
    default_initial_values: ClassVar[dict]

    # the actual unit of `v` is `haldls::vx::CapMemCell::Value`
    # [0–1022]; 1023 means off,
    # but only units included in the `quantity` package are accepted
    units: Final[Dict[str, str]] = {"v": "dimensionless",
                                    "exc_synin": "dimensionless",
                                    "inh_synin": "dimensionless",
                                    "adaptation": "dimensionless"}
    # manual list of all parameters which should not be exposed
    _not_configurable: Final[List[str]] = [
        "event_routing_analog_output",
        "event_routing_enable_digital",
        "leak_reset_i_bias_source_follower",
        "readout_enable_amplifier",
        "readout_source",
        "readout_enable_buffered_access",
        "readout_i_bias"]

    ATOMIC_NEURON_MEMBERS: Final[List[str]] = \
        [name for name, _ in inspect.getmembers(lola.AtomicNeuron())
         if(not(name.startswith("_")) and name.islower())]

    _coco_inject: Optional[Dict[halco.AtomicNeuronOnDLS, lola.AtomicNeuron]]
    # needed to restore after coco was injected
    _user_provided_parameters: Optional[Dict[str, Union[int, bool]]]

    def __init__(self, coco_inject: Optional[dict] = None, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        self._coco_inject = coco_inject
        self._user_provided_parameters = parameters
        super().__init__(**parameters)

    @staticmethod
    # TODO: add more precise return type (cf. feature #3599)
    def get_values(atomic_neuron: lola.AtomicNeuron()) -> dict:
        """Get values of a LoLa Neuron instance as a dict."""

        # TODO: types again like above (cf. feature #3599)
        values = {}

        for member, value in inspect.getmembers(atomic_neuron):
            # skip for non container members
            if member.startswith("_") or not member.islower() \
                    or inspect.ismethod(value) or inspect.isbuiltin(value):
                continue

            for name, inner_value in inspect.getmembers(value):

                # get members
                # exclude lola.AtomicNeuron.EventRouting, since they
                # only have the signature of members, but actually are
                # none
                if name.startswith("_") or not name.islower() \
                    or isinstance(inner_value,
                                  lola.AtomicNeuron.EventRouting):
                    continue
                # asserts just a subset of possible unwanted types
                assert not inspect.ismethod(inner_value)
                assert not inspect.isbuiltin(inner_value)

                key = member + "_" + name
                if key in HXNeuron._not_configurable:
                    continue
                if isinstance(inner_value, bool):
                    values[key] = inner_value
                else:
                    values[key] = float(inner_value)

        return values

    @staticmethod
    def get_default_values() -> dict:
        """Get the default values of a LoLa Neuron."""

        return HXNeuron.get_values(lola.AtomicNeuron())

    @staticmethod
    def _create_translation() -> dict:
        default_values = HXNeuron.get_default_values()
        translation = []
        for key in default_values:
            translation.append((key, key))

        return build_translations(*translation)

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    @staticmethod
    def create_hw_entity(pynn_parameters: dict) -> lola.AtomicNeuron:
        """
        Builds a Lola Neuron with the values from the dict 'pynn_parameters'.
        """

        neuron = lola.AtomicNeuron()
        neuron_members = HXNeuron.ATOMIC_NEURON_MEMBERS

        for param in pynn_parameters:
            member = ""
            cut = 0
            for mem in neuron_members:  # slice
                start_index = param.find(mem)
                if start_index == 0:
                    cut = start_index + len(mem) + 1
                    member = mem
            attr = param[cut:]

            # enable to hand over integers in pynn_parameters
            if isinstance(pynn_parameters[param], numbers.Real):
                pynn_parameters[param] = \
                    int(pynn_parameters[param])

            # set initial values
            real_member = getattr(neuron, member)
            if param == "readout_source":
                setattr(real_member, attr,
                        hal.NeuronConfig.ReadoutSource(
                            pynn_parameters[param]))
            else:
                # PyNN uses lazyarrays for value storage; need to restore
                # original type
                val = type(getattr(real_member,
                                   attr))(pynn_parameters[param])
                setattr(real_member, attr, val)
            setattr(neuron, member, real_member)

        return neuron

    def apply_coco(self, coords: List[halco.AtomicNeuronOnDLS]):
        """
        Extract and apply coco according to provided atomic neuron list

        :param coords: List of coordinates to look up coco. Needs
                       same order and dimensions as parameter_space.
        """
        if self._coco_inject is None:
            # no coco provided -> skip
            return

        param_per_neuron: List[Dict[str, Union[int, bool]]]
        param_per_neuron = []
        param_dict: Dict[str, List[Union[int, bool]]]
        param_dict = {}
        for coord in coords:
            try:
                param_per_neuron.append(
                    self.get_values(self._coco_inject[coord]))
            except KeyError:
                raise KeyError(f"No coco entry for {coord}")
        # "fuse" entries of individual parameters to one large dict of arrays
        for k in param_per_neuron[0].keys():
            param_dict[k] = \
                tuple(coco_entry[k] for coco_entry in param_per_neuron)

        self.parameter_space.update(**param_dict)
        # parameters provided manually by the user have precedence -> overwrite
        self.parameter_space.update(**self._user_provided_parameters)


HXNeuron.default_initial_values = HXNeuron.get_default_values()
HXNeuron.default_parameters = HXNeuron.default_initial_values
# pylint: disable=protected-access
HXNeuron.translations = HXNeuron._create_translation()


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    """
    Spike source, generating spikes according to a Poisson process.
    """

    def __init__(self, start, rate, duration):

        parameters = {"start": start,
                      "rate": rate,
                      "duration": duration}
        super(cells.SpikeSourcePoisson, self).__init__(**parameters)

    translations = build_translations(
        ('start', 'start'),
        ('rate', 'rate'),
        ('duration', 'duration'),
    )

    # Add internal members for the creation of spike trains.
    # 'spike_times' memorizes a fixed Poisson stimulus (for each neuron in the
    # population). 'used_parameters' stores the parameters which were used to
    # generate these spike trains
    _spike_times: Optional[List[np.ndarray]] = None
    _used_parameters: Optional[Dict[str, np.ndarray]] = None

    # TODO: implement L2-based read-out injected spikes
    recordable = []

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    def get_spike_times(self) -> List[np.ndarray]:
        """
        When this function is called for the first time, the spike times for a
        Poisson stimulation are calculated and saved, so that all neurons
        connected to it receive the same stimulation. When a parameter was
        changed (compared to the last calculation of the spike time), the times
        are recalculated.

        :return: (unsorted) spike times for each neuron in the population.
        """

        # generate empty arrays for '_spike_times' and '_used_paramters' if
        # spike times are calculated for the first time
        if self._spike_times is None:
            # Get number of neurons in population from 'start' parameter
            pop_size = self.parameter_space["start"].shape[0]

            self._used_parameters = dict(start=np.zeros(pop_size),
                                         rate=np.zeros(pop_size),
                                         duration=np.zeros(pop_size))
            self._spike_times = [np.array([])] * pop_size

        # Recalculate spike times for which at least one parameter changed
        param_space = self.parameter_space
        change = (param_space["start"] != self._used_parameters["start"]) | \
            (param_space["rate"] != self._used_parameters["rate"]) | \
            (param_space["duration"] != self._used_parameters["duration"])

        for neuron_idx in np.where(change)[0]:
            start = self.parameter_space["start"][neuron_idx]
            rate = self.parameter_space["rate"][neuron_idx]
            duration = self.parameter_space["duration"][neuron_idx]

            rate_in_per_ms = rate / 1000  # Convert rate from 1/s in 1/ms
            num_spikes = np.random.poisson(int(duration * rate_in_per_ms))
            self._spike_times[neuron_idx] = start + \
                np.random.rand(num_spikes) * duration

        # Save parameters which were used to generate the spike trains
        self._used_parameters["start"] = self.parameter_space["start"]
        self._used_parameters["rate"] = self.parameter_space["rate"]
        self._used_parameters["duration"] = self.parameter_space["duration"]

        return self._spike_times


class SpikeSourceArray(cells.SpikeSourceArray):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    # FIXME: workaround, see https://github.com/NeuralEnsemble/PyNN/issues/709
    default_parameters = {'spike_times': ArrayParameter([])}

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )

    # TODO: implement L2-based read-out for injected spikes
    recordable = []

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable
