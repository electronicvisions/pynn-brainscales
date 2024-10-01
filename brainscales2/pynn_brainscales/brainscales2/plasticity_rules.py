from __future__ import annotations
from typing import Dict, Optional, Union, List
from math import ceil
import textwrap
from pyNN.common import Projection, Population
from pynn_brainscales.brainscales2 import simulator
import pygrenade_common as grenade_common
import pygrenade_vx.network as grenade
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

    def to_grenade(
            self, snippet_begin_time: float, snippet_end_time: float) \
            -> grenade.abstract.PlasticityRule.Dynamics.Timer:
        def to_ppu_cycles(value: float) -> int:
            # TODO (Issue #3993): calculate frequency from chip config
            result = float(value)
            result = result * float(hal.Timer.Value.fpga_clock_cycles_per_us)
            result = result * 1000.  # ms -> us
            result = result * 2  # 250MHz vs. 125MHz
            return grenade.abstract.PlasticityRule.Dynamics.Timer.Value(
                int(round(result)))

        timer = grenade.abstract.PlasticityRule.Dynamics.Timer()
        pre_snippet_period_count = ceil(
            max(snippet_begin_time - self.start, 0) / self.period)
        timer.start = to_ppu_cycles(
            self.period * pre_snippet_period_count + self.start)
        timer.period = to_ppu_cycles(self.period)
        timer.num_periods = min(self.num_periods, ceil(
            max(snippet_end_time - self.start, 0) / self.period)
            - pre_snippet_period_count)
        return timer


class PlasticityRule(grenade.abstract.frontend.ExperimentElement):
    """
    Plasticity rule base class.
    Inheritance is to be used for actual implementations.
    Periodic timing information is provided via class `Timer`.
    The kernel implementation is required to be in the form of C++-based PPU
    kernel code.
    """
    _simulator = simulator

    ObservablePerSynapse = grenade.abstract.PlasticityRule\
        .TimedRecordingConfig.ObservablePerSynapse
    ObservablePerNeuron = grenade.abstract.PlasticityRule\
        .TimedRecordingConfig.ObservablePerNeuron
    ObservableArray = grenade.abstract.PlasticityRule\
        .TimedRecordingConfig.ObservableArray

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
        self._recording_data = None
        self.grenade_descriptor = None
        super().__init__(self._simulator.state.grenade_experiment)

    def _set_timer(self, new_timer):
        self._timer = new_timer
        self.changed_input_data = True

    def _get_timer(self):
        self.changed_input_data = True
        return self._timer

    timer = property(_get_timer, _set_timer)

    def _set_observables(self, new_observables):
        self._observables = new_observables
        self.changed_topology = True

    def _get_observables(self):
        self.changed_topology = True
        return self._observables

    observables = property(_get_observables, _set_observables)

    def _set_same_id(self, new_id):
        self._same_id = new_id
        self.changed_topology = True

    def _get_same_id(self):
        self.changed_topology = True
        return self._same_id

    same_id = property(_get_same_id, _set_same_id)

    def _add_projection(self, new_projection: Projection):
        self._projections.append(new_projection)
        self.changed_topology = True

    def _remove_projection(self, old_projection: Projection):
        self._projections.remove(old_projection)
        self.changed_topology = True

    def _add_population(self, new_population: Population):
        self._populations.append(new_population)
        self.changed_topology = True

    def _remove_population(self, old_population: Population):
        self._populations.remove(old_population)
        self.changed_topology = True

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

    def generate_vertex(self) \
            -> grenade.abstract.PlasticityRule:
        recording = None
        if self.observables:
            recording = grenade.abstract.PlasticityRule.TimedRecordingConfig()
            recording.observables = self.observables
        projection_shapes = []
        for proj in self._projections:
            projection_shapes.append(
                grenade_common.CuboidMultiIndexSequence([len(proj)]))
        population_shapes = []
        for pop in self._populations:
            population_shapes.append(
                grenade_common.CuboidMultiIndexSequence(
                    [len(pop)],
                    [grenade_common.CellOnPopulationDimensionUnit()]))
        plasticity_rule = grenade.abstract.PlasticityRule(
            recording,
            grenade.abstract.PlasticityRule.ID(self.same_id),
            population_shapes,
            projection_shapes,
            grenade_common.TimeDomainOnTopology())
        return plasticity_rule

    def generate_dynamics(self, snippet_begin_time, snippet_end_time) \
            -> grenade.abstract.PlasticityRule.Dynamics:
        return grenade.abstract.PlasticityRule.Dynamics(
            timer=self.timer.to_grenade(snippet_begin_time, snippet_end_time),
            batch_size=1)

    def generate_parameterization(self) \
            -> grenade.abstract.PlasticityRule.Parameterization:
        return grenade.abstract.PlasticityRule.Parameterization(
            self.generate_kernel())

    def add_to_topology(
            self,
            experiment: grenade.abstract.frontend.ExperimentSnippet):
        for proj in self._projections:
            if proj.grenade_descriptor is None:
                return False
        for pop in self._populations:
            if pop.grenade_descriptor is None:
                return False

        if self.grenade_descriptor is not None and \
                experiment.topology.contains(
                    self.grenade_descriptor):
            experiment.topology.clear_vertex(self.grenade_descriptor)
            experiment.topology.set(
                self.grenade_descriptor,
                self.generate_vertex())
        else:
            self.grenade_descriptor = experiment.topology.add_vertex(
                self.generate_vertex())

        # Add edges from projections and populations to different ports of
        # plasticity rule
        port = 0
        for proj in self._projections:
            edge = grenade_common.Edge(
                channels_on_source=grenade_common.CuboidMultiIndexSequence(
                    [len(proj)]),
                channels_on_target=grenade_common.CuboidMultiIndexSequence(
                    [len(proj)]),
                port_on_source=1, port_on_target=port)

            experiment.topology.add_edge(
                proj.grenade_descriptor,
                self.grenade_descriptor,
                edge)

            port += 1
        for pop in self._populations:
            edge = grenade_common.Edge(
                channels_on_source=grenade_common.CuboidMultiIndexSequence(
                    [len(pop), 1, 1],
                    [grenade_common.CellOnPopulationDimensionUnit(),
                     grenade_common.CompartmentOnNeuronDimensionUnit(),
                     grenade.abstract.AtomicNeuronOnCompartmentDimensionUnit()]
                ),
                channels_on_target=grenade_common.CuboidMultiIndexSequence(
                    [len(pop)],
                    [grenade_common.CellOnPopulationDimensionUnit()]),
                port_on_source=1, port_on_target=port)

            experiment.topology.add_edge(
                pop.grenade_descriptor,
                self.grenade_descriptor,
                edge)

            port += 1
        return True

    def add_to_input_data(
            self,
            experiment: grenade.abstract.frontend.ExperimentSnippet,
            snippet_begin_time,
            snippet_end_time):
        dynamics_port_index = len(experiment.topology.get(
            self.grenade_descriptor).get_input_ports()) - 1
        experiment.input_data.ports.set(
            (self.grenade_descriptor, dynamics_port_index),
            self.generate_dynamics(
                snippet_begin_time, snippet_end_time))

        parameterization_port_index = len(experiment.topology.get(
            self.grenade_descriptor).get_input_ports()) - 2
        experiment.input_data.ports.set(
            (self.grenade_descriptor, parameterization_port_index),
            self.generate_parameterization())

    def extract_output_data(
            self,
            experiment: List[grenade.abstract.frontend.ExperimentSnippet]):
        self._recording_data = []
        for snippet in experiment:
            if snippet.output_data.ports.contains(
                    (self.grenade_descriptor, 0)):
                self._recording_data.append(snippet.output_data.ports.get(
                    (self.grenade_descriptor, 0)).data)
            else:
                self._recording_data.append(None)

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

        if self._recording_data is None:
            raise RuntimeError(
                "Plasticity rule observables only available after execution.")
        observable_data = []
        for snippet in self._recording_data:
            if observable in snippet.data_array:
                observable_data.append(
                    snippet.data_array[observable][
                        simulator.state.batch_entry])

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
        self.changed_topology = True

    def _set_plasticity_rule(self, new_plasticity_rule):
        self._plasticity_rule = new_plasticity_rule
        self.changed_topology = True

    def _get_plasticity_rule(self):
        self.changed_topology = True
        return self._plasticity_rule

    plasticity_rule = property(_get_plasticity_rule, _set_plasticity_rule)
