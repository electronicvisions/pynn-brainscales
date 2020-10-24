import inspect
import numbers
from typing import List, Dict, ClassVar
from pyNN.standardmodels import cells, build_translations, StandardCellType
from dlens_vx_v2 import lola, hal


class HXNeuron(StandardCellType):

    # exc_synin, inh_synin and adaptation are technical voltages
    recordable: ClassVar[List[str]] = ["spikes", "v", "exc_synin", "inh_synin",
                                       "adaptation"]
    receptor_types: ClassVar[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: ClassVar[bool] = False
    injectable: ClassVar[bool] = True
    default_initial_values: ClassVar[dict]

    # the actual unit of `v` is `haldls::vx::CapMemCell::Value`
    # [0–1022]; 1023 means off,
    # but only units included in the `quantity` package are accepted
    units: ClassVar[Dict[str, str]] = {"v": "dimensionless",
                                       "exc_synin": "dimensionless",
                                       "inh_synin": "dimensionless",
                                       "adaptation": "dimensionless"}
    # manual list of all parameters which should not be exposed
    _not_configurable: ClassVar[List[str]] = [
        "event_routing_analog_output",
        "event_routing_enable_digital",
        "leak_reset_i_bias_source_follower",
        "readout_enable_amplifier",
        "readout_source",
        "readout_enable_buffered_access",
        "readout_i_bias"]

    ATOMIC_NEURON_MEMBERS: ClassVar[List[str]] = \
        [name for name, _ in inspect.getmembers(lola.AtomicNeuron())
         if(not(name.startswith("_")) and name.islower())]

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
    def lola_from_dict(pynn_parameters: dict) -> lola.AtomicNeuron:
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


HXNeuron.default_initial_values = HXNeuron.get_default_values()
HXNeuron.default_parameters = HXNeuron.default_initial_values
# pylint: disable=protected-access
HXNeuron.translations = HXNeuron._create_translation()


class SpikeSourceArray(cells.SpikeSourceArray):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )

    # TODO: implement L2-based read-out for injected spikes
    recordable = []

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable
