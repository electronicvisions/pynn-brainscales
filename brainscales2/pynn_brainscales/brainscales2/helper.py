from typing import Dict
from pathlib import Path
from dlens_vx_v2 import sta, halco, lola, hxcomm


def coco_from_file(path: str) -> dict:
    """
    Extract coco dict from file dump

    :param path: path to file containing coco dump.
    """
    dumper = sta.DumperDone()
    with open(path, 'rb') as fd:
        data = fd.read()
    sta.from_binary(dumper, data)
    return dict(dumper.tolist())


def filter_atomic_neuron(coco: dict) -> Dict[halco.AtomicNeuronOnDLS,
                                             lola.AtomicNeuron]:
    """
    Filter AtomicNeuron entries from coco dict.

    :param coco: coco list, e.g. returned from coco_from_file.
    """
    atomic_neuron_cocos = {coord: container for (coord, container) in
                           coco.items() if isinstance(
                               coord, halco.AtomicNeuronOnDLS)}
    return atomic_neuron_cocos


def filter_non_atomic_neuron(coco: dict) -> dict:
    """
    Filter all non AtomicNeuron entries from coco dict.

    :param coco: coco list, e.g. returned from coco_from_file.
    """
    other_cocos = {coord: container for (coord, container) in coco.items()
                   if not isinstance(coord, halco.AtomicNeuronOnDLS)}
    return other_cocos


def nightly_calib_path() -> Path:
    """
    Find path for calibration automatically from environment.
    Set by Slurm when allocating resources.
    """
    # TODO: opening a connection is architecturally wrong, cf. issue #3868
    with hxcomm.ManagedConnection() as connection:
        identifier = connection.get_unique_identifier()
        path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
            "latest/spiking_cocolist.bin"
        return Path(path)


def filtered_cocos_from_nightly() -> (dict, dict):
    """
    Extract atomic and non-atomic coco lists from nightly calibration
    """
    coco = coco_from_file(nightly_calib_path())
    atomic_coco = filter_atomic_neuron(coco)
    inject_coco = filter_non_atomic_neuron(coco)
    return atomic_coco, inject_coco
