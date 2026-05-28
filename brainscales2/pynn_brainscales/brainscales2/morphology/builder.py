from pynn_brainscales.brainscales2.morphology.mc_neuron_base import \
    McNeuronAutomaticBase
from pygrenade_vx.network.abstract.multicompartment.morphology_builder import \
    MorphologyBuilder as BuilderBase


class MorphologyBuilder(BuilderBase):
    def __init__(self):
        '''
        Builder for multi-compartment neuron classes.

        Structures the neuron during the building process in nodes that
        form a tree.
        '''
        super().__init__(neuron_type=McNeuronAutomaticBase)
