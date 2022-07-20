from typing import Final, List, Set
from pyNN.standardmodels import synapses, build_translations
from pynn_brainscales.brainscales2 import simulator, plasticity_rules
import pygrenade_vx as grenade


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


class StaticRecordingSynapse(
        StaticSynapse, plasticity_rules.PlasticityRule):
    """
    Synaptic connection with fixed weight and delay.
    """

    observable: Final[List[str]] = ["weights", "correlation_causal",
                                    "correlation_acausal"]

    _simulator = simulator

    def __init__(self, timer: plasticity_rules.Timer, weight: float,
                 observables: Set[str]):
        plasticity_rules.PlasticityRule.__init__(self, timer, observables)
        synapses.StaticSynapse.__init__(self, weight=weight)
        self.observables = observables
        self._grenade_generator = None

    def add_to_network_graph(
            self, builder: grenade.logical_network.NetworkBuilder) \
            -> grenade.logical_network.PlasticityRuleDescriptor:
        observables = set(
            getattr(
                grenade.OnlyRecordingPlasticityRuleGenerator.Observable, obs)
            for obs in self.observables)
        self._grenade_generator = grenade.OnlyRecordingPlasticityRuleGenerator(
            observables)
        plasticity_rule = self._grenade_generator.generate()
        logical_plasticity_rule = grenade.logical_network.PlasticityRule()
        logical_plasticity_rule.projections = [
            grenade.logical_network.ProjectionDescriptor(
                self._simulator.state.projections.index(proj))
            for proj in self._projections]
        logical_plasticity_rule.timer = self.timer.to_grenade()
        logical_plasticity_rule.kernel = plasticity_rule.kernel
        logical_plasticity_rule.recording = plasticity_rule.recording
        return builder.add(logical_plasticity_rule)
