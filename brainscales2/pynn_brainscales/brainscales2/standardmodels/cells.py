import inspect
import numbers
from typing import List, Dict, ClassVar
from pyNN.models import BaseCellType
from pyNN.standardmodels import cells, build_translations
from dlens_vx_v1 import lola, hal


class HXNeuron(BaseCellType):

    recordable: ClassVar[List[str]] = ["spikes", "v"]
    receptor_types: ClassVar[List[str]] = ["excitatory", "inhibitory"]
    conductance_based: ClassVar[bool] = False
    injectable: ClassVar[bool] = True
    default_initial_values: ClassVar[dict]

    # the actual unit of `v` is `haldls::vx::CapMemCell::Value`
    # [0–1022]; 1023 means off,
    # but only units included in the `quantity` package are accepted
    units: ClassVar[Dict[str, str]] = {"v": "dimensionless"}

    ATOMIC_NEURON_MEMBERS: ClassVar[List[str]] = \
        [name for name, _ in inspect.getmembers(lola.AtomicNeuron())
         if(not(name.startswith("_")) and name.islower())]

    @staticmethod
    def _get_leak_reset_values() -> dict:
        default_values = {}
        mem = getattr(lola.AtomicNeuron(), "leak_reset")
        for submember, _ in inspect.getmembers(mem):
            if submember.startswith("_") or not submember.islower():
                continue
            # extra nesting
            nesting = ["leak", "reset"]
            if submember in nesting:
                submem = getattr(mem, submember)
                attributes = []
                for name, value in inspect.getmembers(submem):
                    if name.startswith("_") or not name.islower():
                        continue
                    # asserts just a subset of possible
                    # unwanted types
                    assert not inspect.ismethod(value)
                    assert not inspect.isbuiltin(value)
                    attributes.append(name)
                for attr in attributes:
                    default_val = getattr(submem, attr)
                    key = "leak_reset_" + submember + "_" + attr
                    default_values[key] = default_val
            else:
                default_val = getattr(mem, submember)
                key = "leak_reset_" + submember
                default_values[key] = default_val
        return default_values

    @staticmethod
    # TODO: add more precise return type (cf. feature #3599)
    def get_default_values() -> dict:
        """Get the default values of a LoLa Neuron."""

        # TODO: types again like above (cf. feature #3599)
        default_values = {}

        for member, value in inspect.getmembers(lola.AtomicNeuron()):
            if member.startswith("_") or not member.islower():
                continue
            # asserts just a subset of possible unwanted types
            assert not inspect.ismethod(value)
            assert not inspect.isbuiltin(value)

            # special case due to extra nesting
            if member == "leak_reset":
                default_values.update(HXNeuron._get_leak_reset_values())

            else:
                mem = getattr(lola.AtomicNeuron(), member)
                attributes = []
                for name, inner_value in inspect.getmembers(mem):
                    lola_member = getattr(mem, name)

                    # get members
                    # exclude lola.AtomicNeuron.EventRouting, since they
                    # only have the signature of members, but actually are
                    # none
                    if name.startswith("_") or not name.islower() \
                       or isinstance(lola_member,
                                     lola.AtomicNeuron.EventRouting):
                        continue
                    # asserts just a subset of possible unwanted types
                    assert not inspect.ismethod(inner_value)
                    assert not inspect.isbuiltin(inner_value)
                    attributes.append(name)

                for attr in attributes:
                    default_val = getattr(mem, attr)
                    key = member + "_" + attr
                    default_values[key] = default_val

        return default_values

    def can_record(self, variable: str) -> bool:
        return variable in self.recordable

    @staticmethod
    def _set_lola_leak_reset(initial_values: dict, param: str, attr: str,
                             real_member: str) -> None:
        nesting = False
        param_members = ["leak", "reset"]
        param_member = ""
        cut = 0
        for param_mem in param_members:  # slice
            start_index = attr.find(param_mem)
            if start_index == 0:
                cut = start_index + len(param_mem) + 1
                param_member = param_mem
                nesting = True
        param_attr = attr[cut:]
        if nesting:
            nested_member = getattr(real_member, param_member)
            setattr(nested_member, param_attr,
                    initial_values[param].base_value)
            setattr(real_member, param_member, nested_member)
        else:
            setattr(real_member, attr,
                    initial_values[param].base_value)

    @staticmethod
    def lola_from_dict(initial_values: dict) -> lola.AtomicNeuron:
        """
        Builds a Lola Neuron with the values from the dict 'initial_values'.
        """

        neuron = lola.AtomicNeuron()
        neuron_members = HXNeuron.ATOMIC_NEURON_MEMBERS

        for param in initial_values:
            member = ""
            cut = 0
            for mem in neuron_members:  # slice
                start_index = param.find(mem)
                if start_index == 0:
                    cut = start_index + len(mem) + 1
                    member = mem
            attr = param[cut:]

            # enable to hand over integers in initial_values
            if isinstance(initial_values[param].base_value, numbers.Real):
                initial_values[param].base_value = \
                    int(initial_values[param].base_value)

            # set initial values
            real_member = getattr(neuron, member)
            if member == "leak_reset":
                HXNeuron._set_lola_leak_reset(initial_values, param, attr,
                                              real_member)
            elif param == "readout_source":
                setattr(real_member, attr,
                        hal.NeuronConfig.ReadoutSource(
                            initial_values[param].base_value))
            else:
                setattr(real_member, attr, initial_values[param].base_value)
            setattr(neuron, member, real_member)

        return neuron


HXNeuron.default_initial_values = HXNeuron.get_default_values()
HXNeuron.default_parameters = HXNeuron.default_initial_values


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
