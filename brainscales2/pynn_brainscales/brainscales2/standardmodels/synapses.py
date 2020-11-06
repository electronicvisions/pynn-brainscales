from pyNN.standardmodels import synapses, build_translations
from pynn_brainscales.brainscales2 import simulator


class StaticSynapse(synapses.StaticSynapse):
    """
    Synaptic connection with fixed weight and delay.
    """

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )

    # pylint: disable=no-self-use
    def _get_minimum_delay(self):
        return simulator.state.min_delay
