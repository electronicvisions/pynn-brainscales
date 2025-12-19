from __future__ import annotations
from typing import Dict, Optional, Union
from math import ceil
import textwrap
from pyNN.common import Projection, Population
from pynn_brainscales.brainscales2 import simulator
import pygrenade_vx.network as grenade
import pygrenade_vx.signal_flow as grenade_signal_flow
from dlens_vx_v3 import hal, halco


class Timer:
    """
    Periodic timer information for plasticity rule execution.
    """

    def __init__(self, **parameters):
        self._start = parameters["start"]
        self._period = parameters["period"]
        self._num_periods = parameters["num_periods"]
        self.parameters = {x: param for x, param in parameters.items()
                           if x not in ["start", "period", "num_periods"]}

    def _set_start(self, new_start):
        self._start = new_start

    def _get_start(self):
        return self._start

    def _set_period(self, new_period):
        self._period = new_period

    def _get_period(self):
        return self._period

    def _set_num_periods(self, new_num_periods):
        self._num_periods = new_num_periods

    def _get_num_periods(self):
        return self._num_periods

    start = property(_get_start, _set_start)
    period = property(_get_period, _set_period)
    num_periods = property(_get_num_periods, _set_num_periods)

    def to_grenade(self, snippet_begin_time, snippet_end_time) \
            -> grenade.PlasticityRule.Timer:
        def to_ppu_cycles(value: float) -> int:
            # TODO (Issue #3993): calculate frequency from chip config
            result = float(value)
            result = result * float(hal.Timer.Value.fpga_clock_cycles_per_us)
            result = result * 1000.  # ms -> us
            result = result * 2  # 250MHz vs. 125MHz
            return grenade.PlasticityRule.Timer.Value(int(round(result)))

        timer = grenade.PlasticityRule.Timer()
        pre_snippet_period_count = ceil(
            max(snippet_begin_time - self.start, 0) / self.period)
        timer.start = to_ppu_cycles(
            self.period * pre_snippet_period_count + self.start)
        timer.period = to_ppu_cycles(self.period)
        timer.num_periods = min(self.num_periods, ceil(
            max(snippet_end_time - self.start, 0) / self.period)
            - pre_snippet_period_count)
        return timer


class PlasticityRule:
    """
    Plasticity rule base class.
    Inheritance is to be used for actual implementations.
    Periodic timing information is provided via class `Timer`.
    The kernel implementation is required to be in the form of C++-based PPU
    kernel code.
    """
    _simulator = simulator

    ObservablePerSynapse = grenade.PlasticityRule\
        .TimedRecording.ObservablePerSynapse
    ObservablePerNeuron = grenade.PlasticityRule\
        .TimedRecording.ObservablePerNeuron
    ObservableArray = grenade.PlasticityRule\
        .TimedRecording.ObservableArray

    def __init__(self, timer: Timer,
                 observables: Optional[Dict[str, Union[
                     ObservablePerSynapse, ObservablePerNeuron,
                     ObservableArray]]] = None,
                 same_id: int = 0):
        """
        Create a new plasticity rule with timing information.

        :param timer: Timer object.
        :param same_id: Identifier of same plasticity rule.
            Plasticity rules with equal identifier share their
            state across realtime snippets.
            Currently, in addition, the complete provided kernel
            code is required to be equal.
        """
        self._timer = timer
        if observables is None:
            self._observables = {}
        else:
            self._observables = observables
        self._projections = []
        self._populations = []
        self._simulator.state.plasticity_rules.append(self)
        self._same_id = same_id
        self.changed_since_last_run = True

    def _set_timer(self, new_timer):
        self._timer = new_timer
        self.changed_since_last_run = True

    def _get_timer(self):
        self.changed_since_last_run = True
        return self._timer

    timer = property(_get_timer, _set_timer)

    def _set_observables(self, new_observables):
        self._observables = new_observables
        self.changed_since_last_run = True

    def _get_observables(self):
        self.changed_since_last_run = True
        return self._observables

    observables = property(_get_observables, _set_observables)

    def _set_same_id(self, new_id):
        self._same_id = new_id
        self.changed_since_last_run = True

    def _get_same_id(self):
        self.changed_since_last_run = True
        return self._same_id

    same_id = property(_get_same_id, _set_same_id)

    def _add_projection(self, new_projection: Projection):
        self._projections.append(new_projection)
        self.changed_since_last_run = True

    def _remove_projection(self, old_projection: Projection):
        self._projections.remove(old_projection)
        self.changed_since_last_run = True

    def _add_population(self, new_population: Population):
        self._populations.append(new_population)
        self.changed_since_last_run = True

    def _remove_population(self, old_population: Population):
        self._populations.remove(old_population)
        self.changed_since_last_run = True

    def generate_kernel(self) -> str:
        """
        Generate plasticity rule kernel to be compiled into PPU program.
        The interface to be adhered to is the same as in the empty
        implementation below.
        `PLASTICITY_RULE_KERNEL` is the generic name of the kernel function,
        which will be expanded to a unique implementation-defined name upon
        compilation to allow for multiple kernels.

        :return: PPU-code of plasticity-rule kernel as string.
        """
        return textwrap.dedent("""
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "grenade/vx/ppu/neuron_view_handle.h"

        using namespace grenade::vx::ppu;
        using namespace libnux::vx;

        template <size_t N>
        void PLASTICITY_RULE_KERNEL(
            [[maybe_unused]] std::array<SynapseArrayViewHandle, N>& synapses,
            [[maybe_unused]] std::array<NeuronViewHandle, 0>& neurons)
        {}
        """)

    def add_to_network_graph(self, builder: grenade.NetworkBuilder,
                             snippet_begin_time, snippet_end_time) \
            -> grenade.PlasticityRuleOnNetwork:
        plasticity_rule = grenade.PlasticityRule()
        plasticity_rule.timer = self.timer.to_grenade(
            snippet_begin_time, snippet_end_time)
        if self.observables:
            plasticity_rule.recording = grenade.PlasticityRule\
                .TimedRecording()
            plasticity_rule.recording.observables = self.observables
        plasticity_rule.kernel = self.generate_kernel()
        plasticity_rule.projections = [
            proj.synapse_type.to_plasticity_rule_projection_handle(proj)
            for proj in self._projections
        ]
        plasticity_rule.populations = [
            pop.celltype.to_plasticity_rule_population_handle(pop)
            for pop in self._populations
        ]
        plasticity_rule.id = grenade.PlasticityRule.ID(self.same_id)
        return builder.add(plasticity_rule)

    def get_data(
            self,
            network_graph: grenade.NetworkGraph,
            outputs: grenade_signal_flow.OutputData) \
            -> grenade.PlasticityRule.RecordingData:
        """
        Get synaptic and neuron observables of plasticity rule.

        :param network_graph: Network graph to use for lookup of
                              MADC output vertex descriptor.
        :param outputs: All outputs of a single execution to extract
                        samples from.
        :return: Recording data.
        """

        recording_data = grenade\
            .extract_plasticity_rule_recording_data(
                outputs, network_graph,
                grenade.PlasticityRuleOnNetwork(
                    self._simulator.state.plasticity_rules.index(self)))
        return recording_data

    def get_observable_array(self, observable: str) -> object:
        """
        Get data for an array observable.

        :param observable: Name of observable.
        :return: Array with recorded data. The array's entries are values
            for each timer entry. Each value has a `.data` attribute,
            containing the recorded data. This data is twice the size
            set when initializing the observable, since it is added
            for both top and bottom PPUs.

        :raises RuntimeError: If observable name is not known.
        :raises TypeError: If observable is not an ObservableArray.
        """

        if observable not in self.observables:
            raise RuntimeError(
                "Plasticity rule doesn't have requested observable.")
        if not isinstance(self.observables[observable],
                          PlasticityRule.ObservableArray):
            raise TypeError(
                f"Observable {observable} is not an ObservableArray. "
                "For observables per synapse, use the `get_data` function "
                "of the respective projection.")

        observable_data = []
        for array_observables in self._simulator.state.array_observables:
            if observable in array_observables[
                    self._simulator.state.plasticity_rules.index(self)]:
                observable_data.append(array_observables[
                    self._simulator.state.plasticity_rules.
                    index(self)][observable][0])

        return observable_data


class PlasticityRuleHandle:
    """
    Handle to (shared) plasticity rule.
    Inheritance is to be used for actual implementations of cell types.
    """

    _simulator = simulator

    def __init__(self, plasticity_rule: PlasticityRule = None):
        """
        Create a new handle to a plasticity rule.

        :param plasticity_rule: PlasticityRule instance.
        :param observable_options: Observable options to use in this cell type
                                   instance.
        """
        self._plasticity_rule = plasticity_rule
        self.changed_since_last_run = True

    def _set_plasticity_rule(self, new_plasticity_rule):
        self._plasticity_rule = new_plasticity_rule
        self.changed_since_last_run = True

    def _get_plasticity_rule(self):
        self.changed_since_last_run = True
        return self._plasticity_rule

    plasticity_rule = property(_get_plasticity_rule, _set_plasticity_rule)

    # pylint: disable=invalid-name
    @classmethod
    def to_plasticity_rule_population_handle(cls, population: Population) \
            -> grenade.PlasticityRule.PopulationHandle:
        """
        Convert observable options to population handle of plasticity rule
        to backend representation, when plasticity rule handle is assoiated
        to neuron cell type and used in a population.

        :param population: Population for which to convert
        :return: Representation in grenade
        """
        handle = grenade.PlasticityRule.PopulationHandle()
        handle.descriptor = grenade.PopulationOnNetwork(
            cls._simulator.state.populations.index(population))
        handle.neuron_readout_sources = [
            {halco.CompartmentOnLogicalNeuron(): [None]}
            for i in range(len(population))
        ]
        return handle

    # pylint: disable=invalid-name
    @classmethod
    def to_plasticity_rule_projection_handle(cls, projection: Projection) \
            -> grenade.ProjectionOnNetwork:
        """
        Convert observable options to projection handle of plasticity rule
        to backend representation, when plasticity rule handle is assoiated
        to synapse type and used in a projection.

        Currently no options are available.

        :param projection: Projection for which to convert
        :return: Representation in grenade
        """
        return grenade.ProjectionOnNetwork(
            cls._simulator.state.projections.index(projection))
