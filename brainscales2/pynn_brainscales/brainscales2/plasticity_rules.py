from __future__ import annotations
from typing import Dict, Optional
import textwrap
from pyNN.common import Projection
from pynn_brainscales.brainscales2 import simulator
import pygrenade_vx as grenade
from dlens_vx_v3 import hal


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

    def to_grenade(self) -> grenade.logical_network.PlasticityRule.Timer:
        def to_ppu_cycles(value: float) -> int:
            # TODO (Issue #3993): calculate frequency from chip config
            result = float(value)
            result = result * float(hal.Timer.Value.fpga_clock_cycles_per_us)
            result = result * 1000.  # ms -> us
            result = result * 2  # 250MHz vs. 125MHz
            return grenade.PlasticityRule.Timer.Value(int(round(result)))

        timer = grenade.logical_network.PlasticityRule.Timer()
        timer.start = to_ppu_cycles(self.start)
        timer.period = to_ppu_cycles(self.period)
        timer.num_periods = int(self.num_periods)
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

    class Observable:
        pass

    class ObservablePerSynapse(Observable):
        ElementType = grenade.PlasticityRule.TimedRecording \
            .ObservablePerSynapse.Type
        LayoutPerRow = grenade.PlasticityRule.TimedRecording \
            .ObservablePerSynapse.LayoutPerRow

        def __init__(self, element_type: ElementType,
                     layout_per_row: LayoutPerRow):
            self.element_type = element_type
            self.layout_per_row = layout_per_row

    class ObservableArray(Observable):
        ElementType = grenade.PlasticityRule.TimedRecording \
            .ObservableArray.Type

        def __init__(self, element_type: ElementType, size: int):
            self.element_type = element_type
            self.size = size

    def __init__(self, timer: Timer,
                 observables: Optional[Dict[str, Observable]] = None):
        """
        Create a new plasticity rule with timing information.

        :param timer: Timer object.
        """
        self._timer = timer
        if observables is None:
            self._observables = {}
        else:
            self._observables = observables
        self._projections = []
        self._simulator.state.plasticity_rules.append(self)
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

    def _add_projection(self, new_projection: Projection):
        self._projections.append(new_projection)
        self.changed_since_last_run = True

    def _remove_projection(self, old_projection: Projection):
        self._projections.remove(old_projection)
        self.changed_since_last_run = True

    # pylint: disable=no-self-use
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
        #include "libnux/vx/location.h"
        #include "grenade/vx/ppu/synapse_array_view_handle.h"

        using namespace grenade::vx::ppu;
        using namespace libnux::vx;

        template <size_t N>
        void PLASTICITY_RULE_KERNEL(
            [[maybe_unused]] std::array<SynapseArrayViewHandle, N>& synapses,
            [[maybe_unused]] std::array<PPUOnDLS, N> const& synrams)
        {}
        """)

    def add_to_network_graph(self, builder: grenade.logical_network
                             .NetworkBuilder) \
            -> grenade.logical_network.PlasticityRuleDescriptor:
        plasticity_rule = grenade.logical_network.PlasticityRule()
        plasticity_rule.timer = self.timer.to_grenade()
        if self.observables:
            plasticity_rule.recording = grenade.PlasticityRule.TimedRecording()
            observables = {}
            for name, observable in self.observables.items():
                if isinstance(observable, self.ObservablePerSynapse):
                    grenade_observable = grenade.PlasticityRule \
                        .TimedRecording.ObservablePerSynapse()
                    grenade_observable.layout_per_row = observable \
                        .layout_per_row
                elif isinstance(observable, self.ObservableArray):
                    grenade_observable = grenade.PlasticityRule \
                        .TimedRecording.ObservableArray()
                    grenade_observable.size = observable.size
                else:
                    raise RuntimeError("Observable type not implemented.")
                grenade_observable.type = observable.element_type
                observables.update({name: grenade_observable})
            plasticity_rule.recording.observables = observables
        plasticity_rule.kernel = self.generate_kernel()
        plasticity_rule.projections = [
            grenade.logical_network.ProjectionDescriptor(
                self._simulator.state.projections.index(proj))
            for proj in self._projections
        ]
        return builder.add(plasticity_rule)

    def get_data(
            self,
            logical_network_graph: grenade.logical_network.NetworkGraph,
            hardware_network_graph: grenade.NetworkGraph,
            outputs: grenade.IODataMap) -> grenade.logical_network \
            .PlasticityRule.RecordingData:
        """
        Get synaptic observables of plasticity rule.
        :param network_graph: Network graph to use for lookup of
                              MADC output vertex descriptor
        :param outputs: All outputs of a single execution to extract
                        samples from
        :return: Recording data
        """

        recording_data = grenade.logical_network\
            .extract_plasticity_rule_recording_data(
                outputs,
                logical_network_graph, hardware_network_graph,
                grenade.logical_network.PlasticityRuleDescriptor(
                    self._simulator.state.plasticity_rules.index(self)))
        return recording_data
