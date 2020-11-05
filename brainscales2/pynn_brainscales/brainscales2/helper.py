from typing import Dict
from dlens_vx_v2 import sta, halco, lola


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
