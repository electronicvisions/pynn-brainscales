#!/usr/bin/env python

import unittest
import numpy as np
import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse


class TestPoissonInput(unittest.TestCase):

    def setUp(self):
        self.bg_props = dict(
            start=1,  # ms
            rate=20e3,  # Hz
            duration=100  # ms
        )
        # Emulate the network 1ms longer than Poisson stimulation, in order to
        # convince oneself that the stimulation ends properly.
        self.runtime = self.bg_props["start"] + self.bg_props["duration"] + 1

        # The refractory time was found to must be set slightly larger 0 (DAC
        # value) to achieve a short time on hardware (cf. #3741).
        neuron_params = {"refractory_period_refractory_time": 5}

        # By enabling the bypass mode, the neuron should spike at the income of
        # each event.
        pynn.setup(enable_neuron_bypass=True)

        pop1 = pynn.Population(1, pynn.cells.HXNeuron(**neuron_params))
        pop2 = pynn.Population(1, pynn.cells.HXNeuron(**neuron_params))
        pop1.record(["spikes"])
        pop2.record(["spikes"])
        src = pynn.Population(
            2, pynn.cells.SpikeSourcePoisson(**self.bg_props))
        src.record(["spikes"])
        # The second Poisson neuron is not connected to any target population
        # and just exists to test that Populations with more than one neuron
        # are created correctly.

        pynn.Projection(src, pop1, pynn.OneToOneConnector(),
                        synapse_type=StaticSynapse(weight=63))
        pynn.Projection(src, pop2, pynn.OneToOneConnector(),
                        synapse_type=StaticSynapse(weight=63))

        pynn.run(self.runtime)

        self.spiketrain1 = pop1.get_data("spikes").segments[0].spiketrains[0]
        self.spiketrain2 = pop2.get_data("spikes").segments[0].spiketrains[0]
        self.spiketrain_src = src.get_data("spikes").segments[0].spiketrains[0]
        pynn.reset()

        self.rate3 = 10e3
        src.set(rate=self.rate3)
        pynn.run(self.runtime)
        self.spiketrain3 = pop1.get_data("spikes").segments[1].spiketrains[0]
        pynn.end()

    def test_timeframe(self):
        """
        Check that the neurons don't spike before 'start' or after
        'start' + 'duration'.
        """
        self.assertGreaterEqual(self.spiketrain1[0], self.bg_props["start"])
        self.assertLessEqual(self.spiketrain1[-1], self.bg_props["start"]
                             + self.bg_props["duration"])

    def test_length(self):
        """
        Check if neuron spikes as often as expected by the stated background
        properties.
        """
        expected_length1 = self.bg_props["duration"] / 1000 * \
            self.bg_props["rate"]
        dev1 = 5 * np.sqrt(expected_length1)
        measured_length1 = len(self.spiketrain1)
        self.assertLessEqual(abs(measured_length1 - expected_length1), dev1)

        measured_length_src = len(self.spiketrain_src)
        self.assertLessEqual(abs(measured_length_src - expected_length1), dev1)

        expected_length3 = self.bg_props["duration"] / 1000 * self.rate3
        dev3 = 5 * np.sqrt(expected_length3)
        measured_length3 = len(self.spiketrain3)
        self.assertLessEqual(abs(measured_length3 - expected_length3), dev3)

    def test_equality(self):
        """
        Check that two neurons receive the same Poisson stimulation. At least
        98% (estimate considering typical spike loss) of the spikes are read
        back and that the time difference between them is less than 0.01ms.
        """
        # Add a small number such that timestamps with a '5' the third decimal
        # place are rounded up to the next higher (and not to the next even)
        # number.
        spiketrain1_set = set(np.round(self.spiketrain1.magnitude + 1e-9, 2))
        spiketrain2_set = set(np.round(self.spiketrain2.magnitude + 1e-9, 2))
        spiketrain_src_set = set(np.round(
            self.spiketrain_src.magnitude + 1e-9, 2))
        self.assertLess(abs(len(spiketrain1_set - spiketrain2_set))
                        / len(spiketrain1_set),
                        0.02)
        self.assertLess(abs(len(spiketrain1_set - spiketrain_src_set))
                        / len(spiketrain1_set),
                        0.02)

    def test_poisson(self):
        """
        Check that the power spectrum of the spiketrain follows a Delta
        distribution around 0 Hz as expected for a Poisson process.
        """
        # calculate the power spectrum
        n_bins = self.runtime * 1000  # = 1ms bin in bio
        hist, edges = np.histogram(self.spiketrain1, bins=n_bins)
        p_spec = np.abs(np.fft.fft(hist))**2
        freq = np.fft.fftfreq(len(hist), edges[1] - edges[0])

        # visualize power spectrum
        fig, ax = plt.subplots()
        ax.plot(freq, p_spec, linestyle="None", marker=".")
        ax.set_xlabel(r"frequency [$\frac{1}{ms}$]")
        ax.set_ylabel("power spectral density")
        ax.set_yscale("log")
        fig.savefig("plot_poisson_power_spectrum.pdf")

        # frequency 0 Hz is at index 0
        assert np.where(freq == 0)[0].item() == 0

        # Since the power spectral density is distributed over multiple orders
        # of magnitude, the log is considered.
        log_band = np.log(p_spec[1:])

        # The power spectral density everywhere except at 0 Hz is expected to
        # be distributed randomly within a narrow band around a mean small
        # compared to the Delta peak.
        p_spec_mean = np.mean(log_band)
        p_spec_std = np.std(log_band)
        p_spec_max = np.max(log_band)

        # The random band should be completely smaller than a critical value,
        # which depends on the runtime.
        limit_band_abs = np.log(1e5)
        self.assertLess(p_spec_max, limit_band_abs)

        # No side peaks are expected, so the highest spectral density of the
        # random band should not exceed the mean significantly.
        limit_band_rel = 5 * p_spec_std
        self.assertLess(p_spec_max - p_spec_mean, limit_band_rel)

        # However the Delta peak at 0 Hz should exceed the random band
        # significantly.
        # The significant distance again depends on the runtime and the value
        # chosen here is found to be appropiate for a runtime of 100ms.
        significant_dist = 1e2
        self.assertGreater(p_spec[0] / np.max(p_spec[1:]), significant_dist)


if __name__ == "__main__":
    unittest.main()
