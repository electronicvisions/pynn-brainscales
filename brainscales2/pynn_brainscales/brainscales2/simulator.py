import numpy as np
from pyNN.common import IDMixin, Population
from pyNN.common.control import BaseState
from pynn_brainscales.brainscales2.standardmodels.cells import HXNeuron
from dlens_vx import hal, halco, sta, hxcomm


name = "HX"  # for use in annotating output data


class ID(int, IDMixin):
    __doc__ = IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""

        cell_id_size = halco.AtomicNeuronOnDLS.size  # 512
        assert isinstance(cell_id_size, int)
        if n < cell_id_size:
            int.__init__(n)
            IDMixin.__init__(self)
        else:
            raise ValueError("Maximal number of neurons is {}.".format(
                cell_id_size))


class _State(BaseState):
    """Represent the simulator state."""

    # pragma pylint: disable=invalid-name
    def __init__(self):
        super(_State, self).__init__()

        self.spikes = []
        self.times = []
        self.membrane = []

        self.mpi_rank = 0        # disabled
        self.num_processes = 1   # number of MPI processes
        self.t = 0
        # TODO: replace by calculation (cf. feature #3594)
        self.dt = 3.4e-05        # average time between two MADC samples
        self.populations = []
        self.recorders = set([])
        self.clear()

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    @staticmethod
    def get_spikes(spikes: np.ndarray, runtime: int) -> np.ndarray:
        label = spikes["label"]
        spiketime = spikes["chip_time"] \
            / (int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        label = label[(spiketime <= runtime)]
        spiketime = spiketime[spiketime <= runtime]
        return_spikes = np.array((label, spiketime)).T
        return return_spikes

    @staticmethod
    def get_v(v_input: np.ndarray) -> [np.ndarray, np.ndarray]:
        membrane = v_input["value"][1:]
        times = v_input["chip_time"][1:] \
            / (int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1000)
        return times, membrane

    @staticmethod
    def configure_common(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:

        # set i_bias_threshold_comparator for all CapMemBlock
        for coord in halco.iter_all(halco.CapMemBlockOnDLS):
            builder.write(coord, hal.CapMemBlock())
            conf = halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.
                                         neuron_i_bias_threshold_comparator,
                                         block=coord)
            builder.write(conf, hal.CapMemCell(500))

        # disables event recording during configuration time
        rec_config = hal.EventRecordingConfig()
        rec_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        # set all neurons on chip to default values
        default_neuron = HXNeuron.lola_from_dict({})
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            builder.write(coord, default_neuron)

        config = hal.CommonNeuronBackendConfig()
        config.clock_scale_fast = 3
        config.clock_scale_slow = 3
        config.enable_clocks = True
        config.enable_event_registers = True
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)
        return builder

    # pylint: disable=unsupported-assignment-operation
    @staticmethod
    def configure_madc(builder: sta.PlaybackProgramBuilder,
                       coord: halco.AtomicNeuronOnDLS) \
            -> sta.PlaybackProgramBuilder:
        """Connect MADC to neuron on given coord."""

        hemisphere = coord.toNeuronRowOnDLS().toHemisphereOnDLS()
        is_odd = (int(coord.toNeuronColumnOnDLS()) % 2) == 1
        is_even = not is_odd

        config = hal.ReadoutSourceSelection()
        smux = config.SourceMultiplexer()
        smux.neuron_odd[halco.HemisphereOnDLS(hemisphere)] = is_odd
        smux.neuron_even[halco.HemisphereOnDLS(hemisphere)] = is_even
        config.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0), smux)
        config.enable_buffer_to_pad[
            halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        # TODO: move to Recorder class (cf. feature #3595)
        # set capmem cells
        readout_params = {
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_0: 0,
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_1: 0,
            halco.CapMemCellOnDLS.readout_pseudo_diff_buffer_bias: 0,
            halco.CapMemCellOnDLS.readout_ac_mux_i_bias: 500,
            halco.CapMemCellOnDLS.readout_madc_in_500na: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_i_bias: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_v_ref: 400,
            halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref: 400,
            halco.CapMemCellOnDLS.readout_iconv_test_voltage: 400,
            halco.CapMemCellOnDLS.readout_iconv_sc_amp_v_ref: 400,
        }

        for key, variable in readout_params.items():
            builder.write(key, hal.CapMemCell(variable))

        return builder

    @staticmethod
    def configure_population(builder: sta.PlaybackProgramBuilder,
                             population: Population,
                             enable_spike_recording: bool,
                             enable_v_recording: bool) \
            -> sta.PlaybackProgramBuilder:
        """
        Places Neurons in Population "pop" on chip and configures spike and
        v recording.
        """

        # checks if the celltype is compatible with the hardware
        assert isinstance(population.celltype, HXNeuron)

        # configures a neuron with the passed initial values
        atomic_neuron = HXNeuron.lola_from_dict(population.initial_values)

        # places the neurons from pop on chip
        for coord in population.all_cells:
            coord_int = coord
            coord = halco.AtomicNeuronOnDLS(
                halco.AtomicNeuronOnDLS.enum_type(coord))

            # configure spike recording
            if enable_spike_recording:
                atomic_neuron.event_routing.enable_analog = True
                atomic_neuron.event_routing.enable_digital = True
                # TODO: enhance address setting (cf. feature #3596)
                atomic_neuron.event_routing.address = coord_int
                atomic_neuron.leak_reset.i_bias_source_follower = 280

            # configure v recording
            if enable_v_recording:
                atomic_neuron.event_routing.enable_analog = True
                atomic_neuron.event_routing.enable_digital = True
                atomic_neuron.leak_reset.i_bias_source_follower = 280
                atomic_neuron.readout.enable_amplifier = True
                atomic_neuron.readout.enable_buffered_access = True
                atomic_neuron.readout.i_bias = 1000
                builder = state.configure_madc(builder, coord)

            builder.write(coord, atomic_neuron)

        return builder

    @staticmethod
    def madc_configuration(builder: sta.PlaybackProgramBuilder,
                           runtime: float) \
            -> sta.PlaybackProgramBuilder:
        """Configure number of MADC samples depending on set runtime."""
        config = hal.MADCConfig()
        config.number_of_samples = int(runtime / state.dt)
        builder.write(halco.MADCConfigOnDLS(), config)
        return builder

    @staticmethod
    def madc_start_recording(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """Start MADC recording."""
        config = hal.MADCControl()
        config.enable_power_down_after_sampling = True
        config.start_recording = False
        config.wake_up = True
        config.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), config)
        return builder

    @staticmethod
    def madc_stop_recording(builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """Stop MADC recording."""
        config = hal.MADCControl()
        config.enable_power_down_after_sampling = True
        config.start_recording = True
        config.wake_up = False
        config.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), config)
        return builder

    @staticmethod
    def run_on_chip(builder: sta.PlaybackProgramBuilder, runtime: float,
                    v_recording: bool) \
            -> sta.PlaybackProgramBuilder:
        """
        Runs the experiment on chip and records MADC samples, if there is a
        recorder recording 'v'.
        """

        # enable event recording
        rec_config = hal.EventRecordingConfig()
        rec_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(0), rec_config)

        if v_recording:
            builder = state.madc_start_recording(builder)

        # wait 100 us to buffer some program in FPGA to reach precise timing
        # afterwards and sync time
        initial_wait = 100  # us
        builder.write(halco.TimerOnDLS(), hal.Timer())
        builder.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())

        if v_recording:
            builder = state.madc_stop_recording(builder)

        # record for time 'runtime'
        builder.block_until(halco.TimerOnDLS(), hal.Timer.Value(
            int(int(hal.Timer.Value.fpga_clock_cycles_per_us)
                * ((runtime * 1000) + initial_wait))))

        return builder

    def run(self, runtime):
        self.t += runtime
        self.running = True

        # initialize chip
        builder1, _ = sta.ExperimentInit().generate()

        # common configuration
        builder1 = self.configure_common(builder1)

        # configure populations
        configured = []
        v_recording = False
        for recorder in self.recorders:
            population = recorder.population
            configured.append(population)
            enable_spike_recording = False
            enable_v_recording = False
            for variable in recorder.recorded:
                if variable == "spikes":
                    enable_spike_recording = True
                elif variable == "v":
                    enable_v_recording = True
                    v_recording = True
                else:
                    raise NotImplementedError
            builder1 = self.configure_population(
                builder1,
                population,
                enable_spike_recording=enable_spike_recording,
                enable_v_recording=enable_v_recording)

        if v_recording:
            builder1 = self.madc_configuration(builder1, runtime)

        # wait 20000 us for capmem voltages to stabilize
        initial_wait = 20000  # us
        builder1.write(halco.TimerOnDLS(), hal.Timer())
        builder1.block_until(halco.TimerOnDLS(), int(
            initial_wait * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

        builder2 = sta.PlaybackProgramBuilder()
        builder2 = self.run_on_chip(builder2, runtime, v_recording)

        # TODO: add stats readout (cf. feature #3597)
        program = builder2.done()
        with hxcomm.ManagedConnection() as conn:
            sta.run(conn, builder1.done())
            sta.run(conn, program)

        # make list 'spikes' of tupel (neuron id, spike time)
        self.spikes = self.get_spikes(program.spikes.to_numpy(), runtime)

        # make two list for madc samples: times, membrane
        self.times, self.membrane = self.get_v(program.madc_samples.to_numpy())


state = _State()  # a Singleton, so only a single instance ever exists
