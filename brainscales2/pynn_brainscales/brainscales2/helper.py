import urllib
from pathlib import Path
from dlens_vx_v3 import sta, hxcomm


def chip_from_portable_binary(data: bytes) -> dict:
    """
    Convert portable binary data to chip object.

    :param data: Coco list in portable binary format.
    :return: lola chip configuration.
    """
    dumper = sta.DumperDone()
    sta.from_portablebinary(dumper, data)
    return sta.convert_to_chip(dumper)


def chip_from_file(path: str) -> dict:
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


def chip_from_nightly() -> (dict):
    """
    Extract chip config from nightly calibration.
    """

    # First check local filesystem if calibration does not exist try to
    # download it
    if nightly_calib_path().exists():
        chip = chip_from_file(nightly_calib_path())
    else:
        try:
            data = urllib.request.urlopen(nightly_calib_url()).read()
            chip = chip_from_portable_binary(data)

        except urllib.error.URLError:
            raise RuntimeError('Could not find a nightly calibration for '
                               f'setup "{get_unique_identifier()}".')

    return chip
