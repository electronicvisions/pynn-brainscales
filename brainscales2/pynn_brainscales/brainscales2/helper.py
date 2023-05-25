from typing import List, Optional, Tuple
import inspect
import urllib
from pathlib import Path
from dlens_vx_v3 import sta, hxcomm, lola


def chip_from_portable_binary(data: bytes) -> lola.Chip:
    """
    Convert portable binary data to chip object.

    :param data: Coco list in portable binary format.
    :return: lola chip configuration.
    """
    dumper = sta.DumperDone()
    sta.from_portablebinary(dumper, data)
    return sta.convert_to_chip(dumper)


def chip_from_file(path: Path) -> lola.Chip:
    """
    Extract chip config from coco file dump

    :param path: path to file containing coco dump.
    """
    with open(path, 'rb') as fd:
        data = fd.read()
    return chip_from_portable_binary(data)


def get_unique_identifier() -> str:
    """
    Retrieve the unique identifier of the current chip.

    Set by Slurm when allocating resources.
    """
    # TODO: opening a connection is architecturally wrong, cf. issue #3868
    with hxcomm.ManagedConnection() as connection:
        identifier = connection.get_unique_identifier()
    return identifier


def nightly_calib_path() -> Path:
    """
    Find path for nightly calibration.
    """
    identifier = get_unique_identifier()
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        "latest/spiking_cocolist.pbin"
    return Path(path)


def nightly_calib_url() -> str:
    """
    Find url for nightly calibration.
    """
    identifier = get_unique_identifier()
    return "https://openproject.bioai.eu/data_calibration/" \
           f"hicann-dls-sr-hx/{identifier}/stable/latest/" \
           "spiking_cocolist.pbin"


def chip_from_nightly() -> lola.Chip:
    """
    Extract chip config from nightly calibration.
    """

    # First check local filesystem if calibration does not exist try to
    # download it
    if nightly_calib_path().exists():
        chip = chip_from_file(nightly_calib_path())
    else:
        try:
            # urllib handles closing on destruction of data
            # pylint: disable=consider-using-with
            data = urllib.request.urlopen(nightly_calib_url()).read()
            chip = chip_from_portable_binary(data)

        except urllib.error.URLError as ex:
            raise RuntimeError('Could not find a nightly calibration for '
                               f'setup "{get_unique_identifier()}".') from ex

    return chip


# TODO: add more precise return type (cf. feature #3599)
def get_values_of_atomic_neuron(atomic_neuron: lola.AtomicNeuron(),
                                exclude: Optional[List[str]] = None) -> dict:
    """
    Get values of a LoLa Neuron instance as a dict.

    Parse the atomic neuron and save the values of all members and their
    attributes in a dictionary. The keys of the dictionary are the member's
    name and it's attributes name combined by an underscore.

    :param atomic_neuron: Atomic neuron from which to get the values.
    :param exclude: Members to exclude from parsing.
    :return: Dictionary with the values saved in the atomic neuron.
    """

    if exclude is None:
        exclude = []

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
            if key in exclude:
                continue
            if isinstance(inner_value, bool):
                values[key] = inner_value
            else:
                values[key] = float(inner_value)

    return values


def decompose_in_member_names(composed_name: str) -> Tuple[str, str]:
    '''
    Extract member and attribute of a lola.AtomicNeuron from composed name.

    This function can be used to retrieve the original name of the member and
    the attributes of the `lola.AtomicNeuron` for the keys in the dictionary
    returned by `get_values_of_atomic_neuron`.
    For example 'multicompartment_i_bias_nmda' is decomposed in the member
    'multicompartment' and  the attribute 'i_bias_nmda'.

    :param composed_name: String which is a combination of a member of the
        lola.AtomicNeuron and an attribute of it. The two are combined by
        an underscore.
    :return: Tuple of the name of the member and the attribute.
    '''
    member = ""
    cut = 0
    for mem in ATOMIC_NEURON_MEMBERS:  # slice
        start_index = composed_name.find(mem)
        if start_index == 0:
            cut = start_index + len(mem) + 1
            member = mem
    attr = composed_name[cut:]

    return member, attr


# all members of the lola.AtomicNeuron
ATOMIC_NEURON_MEMBERS = \
    [name for name, _ in inspect.getmembers(lola.AtomicNeuron())
     if(not(name.startswith("_")) and name.islower())]
