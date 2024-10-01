from typing import Final, List, Set
from pyNN.standardmodels import synapses, build_translations
from pynn_brainscales.brainscales2 import simulator, plasticity_rules
import pygrenade_vx.network.abstract as grenade


class StaticSynapse(synapses.StaticSynapse):
    """
    Synaptic connection with fixed weight and delay.
    """

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )

    def _get_minimum_delay(self):
        return simulator.state.min_delay


class PlasticSynapse(
        synapses.StaticSynapse,
        plasticity_rules.PlasticityRuleHandle):
    """
    Synaptic connection with fixed initial weight and delay and handle to
    plasticity rule.
    """

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )

    def __init__(
            self, weight: int = 0, delay: float = 0,
            plasticity_rule: plasticity_rules.PlasticityRule = None):
        synapses.StaticSynapse.__init__(
            self, weight=weight, delay=delay)
        plasticity_rules.PlasticityRuleHandle.__init__(
            self, plasticity_rule=plasticity_rule)


class StaticRecordingSynapse(
        StaticSynapse, plasticity_rules.PlasticityRuleHandle):
    """
    Synaptic connection with fixed weight and delay.
    """

    class RecordingRule(plasticity_rules.PlasticityRule):
        """
        "Plasticity" rule only usable for generating recording of observables.
        Reference to set of observables is stored and used.
        """
        def __init__(
                self, timer: plasticity_rules.Timer,
                observables: Set[str]):
            plasticity_rules.PlasticityRule.__init__(
                self, timer=timer, observables=None)
            self._recording_observables = observables

        def _get_observables(self):
            observables = set(
                getattr(
                    grenade.OnlyRecordingPlasticityRuleGenerator.Observable,
                    obs)
                for obs in self._recording_observables)
            grenade_generator = grenade\
                .OnlyRecordingPlasticityRuleGenerator(observables)
            return grenade_generator.generate()[0].observables

        def _set_observables(self, new_observables):
            raise RuntimeError(
                "Setting observables not possible directly, "
                "use observables on synapse type instead.")

        observables = property(_get_observables, _set_observables)

        def generate_vertex(self) -> grenade.PlasticityRule:
            vertex = super().generate_vertex()
            observables = set(
                getattr(
                    grenade
                    .OnlyRecordingPlasticityRuleGenerator.Observable,
                    obs)
                for obs in self._recording_observables)
            grenade_generator = grenade\
                .OnlyRecordingPlasticityRuleGenerator(observables)
            vertex.recording = grenade_generator.generate()[0]
            return vertex

        def generate_kernel(self) -> str:
            observables = set(
                getattr(
                    grenade
                    .OnlyRecordingPlasticityRuleGenerator.Observable,
                    obs)
                for obs in self._recording_observables)
            grenade_generator = grenade\
                .OnlyRecordingPlasticityRuleGenerator(observables)
            return grenade_generator.generate()[1].kernel

    POSSIBLE_OBSERVABLES: Final[List[str]] = [
        "weights", "correlation_causal", "correlation_acausal"]

    def __init__(self, timer: plasticity_rules.Timer, weight: float,
                 observables: Set[str]):
        # pylint: disable=super-init-not-called
        self._observables = observables
        plasticity_rules.PlasticityRuleHandle.__init__(
            self, self.RecordingRule(timer, self._observables))
        StaticSynapse.__init__(self, weight=weight)
        self.changed_topology = True

    def _get_observables(self):
        self.changed_topology = True
        return self._observables

    def _set_observables(self, value: Set[str]):
        self.changed_topology = True
        self._observables = value
        super().plasticity_rule._recording_observables = self._observables  # pylint: disable=protected-access

    observables = property(_get_observables, _set_observables)

    def _get_plasticity_rule(self):
        return super().plasticity_rule

    def _set_plasticity_rule(self, new_plasticity_rule):
        raise RuntimeError(
            "Plasticity rule assignment not possible for "
            "StaticRecordingSynapse.")

    plasticity_rule = property(_get_plasticity_rule, _set_plasticity_rule)
