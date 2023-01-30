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
            return grenade_generator.generate().recording.observables

        def _set_observables(self, value):
            raise RuntimeError(
                "Setting observables not possible directly, "
                "use observables on synapse type instead.")

        observables = property(_get_observables, _set_observables)

        def add_to_network_graph(
                self, builder: grenade.logical_network.NetworkBuilder) \
                -> grenade.logical_network.PlasticityRuleDescriptor:
            observables = set(
                getattr(
                    grenade.OnlyRecordingPlasticityRuleGenerator.Observable,
                    obs)
                for obs in self._recording_observables)
            grenade_generator = grenade\
                .OnlyRecordingPlasticityRuleGenerator(observables)
            plasticity_rule = grenade_generator.generate()
            logical_plasticity_rule = grenade.logical_network.PlasticityRule()
            logical_plasticity_rule.projections = [
                grenade.logical_network.ProjectionDescriptor(
                    self._simulator.state.projections.index(proj))
                for proj in self._projections]
            logical_plasticity_rule.timer = self.timer.to_grenade()
            logical_plasticity_rule.kernel = plasticity_rule.kernel
            logical_plasticity_rule.recording = plasticity_rule.recording
            return builder.add(logical_plasticity_rule)

    POSSIBLE_OBSERVABLES: Final[List[str]] = [
        "weights", "correlation_causal", "correlation_acausal"]

    def __init__(self, timer: plasticity_rules.Timer, weight: float,
                 observables: Set[str]):
        self._observables = observables
        plasticity_rules.PlasticityRuleHandle.__init__(
            self, self.RecordingRule(timer, self._observables))
        synapses.StaticSynapse.__init__(self, weight=weight)
        self.changed_since_last_run = True

    def _get_observables(self):
        self.changed_since_last_run = True
        return self._observables

    def _set_observables(self, value: Set[str]):
        self.changed_since_last_run = True
        self._observables = value
        super().plasticity_rule._recording_observables = self._observables

    observables = property(_get_observables, _set_observables)

    def _get_plasticity_rule(self):
        return super().plasticity_rule

    def _set_plasticity_rule(self, value):
        raise RuntimeError(
            "Plasticity rule assignment not possible for "
            "StaticRecordingSynapse.")

    plasticity_rule = property(_get_plasticity_rule, _set_plasticity_rule)
