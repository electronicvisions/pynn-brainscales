#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


class TestCalibHXNeuronCuba(unittest.TestCase):

    def setUp(self):
        pynn.setup()
        self.pop = pynn.Population(3, pynn.cells.CalibHXNeuronCuba())
        self.pop.record(["spikes"])

    def tearDown(self):
        pynn.end()

    def test_empty(self):
        """
        Default calibrated neurons should be silent without excitation
        """
        pynn.run(10)
        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertEqual(len(spiketrain), 0)

    def test_external_stimulus(self):
        """
        Default calibrated neurons should fire with enough excitation
        """
        exc_stim_pop = pynn.Population(
            10, pynn.standardmodels.cells.SpikeSourcePoisson(
                rate=1e5, start=0, duration=10))

        pynn.Projection(
            exc_stim_pop, self.pop, pynn.AllToAllConnector(),
            synapse_type=pynn.standardmodels.synapses.StaticSynapse(weight=63),
            receptor_type="excitatory")
        pynn.run(10)

        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertGreater(len(spiketrain), 0)

    def test_leak_over_threshold(self):
        """
        Manually setting leak to max should lead to firing of the neuron.
        """
        pynn.preprocess()
        self.pop.actual_hwparams[0].leak.v_leak = 1022
        pynn.run(10)

        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertGreater(len(spiketrain), 0)

    def test_access_calib_result(self):
        """
        Calibration related parameters only available after run() or
        preprocess().
        """
        with self.assertRaises(AttributeError):
            _ = self.pop.calib_target
        with self.assertRaises(AttributeError):
            _ = self.pop.calib_hwparams
        with self.assertRaises(AttributeError):
            _ = self.pop.actual_hwparams

        pynn.preprocess()
        _ = self.pop.calib_target
        _ = self.pop.calib_hwparams
        _ = self.pop.actual_hwparams

    def test_popview(self):
        """
        Test correct assignment/masking in population views.
        """
        pynn.preprocess()
        # modifying second entry in the population view should change the third
        # neuron in the parent population
        self.pop[1:3].actual_hwparams[1].leak.v_leak = 1022
        self.assertEqual(self.pop.actual_hwparams[2].leak.v_leak, 1022)
        self.assertNotEqual(self.pop.actual_hwparams[0].leak.v_leak, 1022)
        self.assertNotEqual(self.pop.actual_hwparams[1].leak.v_leak, 1022)

    def test_non_default_params(self):
        """
        Test access to all parameters.
        """
        default = self.pop.celltype.default_parameters
        self.pop.set(
            v_rest=default["v_rest"] - 1,
            v_reset=default["v_reset"] - 1,
            v_thresh=default["v_thresh"] - 1,
            tau_m=default["tau_m"] - 1,
            tau_syn_E=default["tau_syn_E"] - 1,
            tau_syn_I=default["tau_syn_I"] - 1,
            cm=default["cm"] - 1,
            tau_refrac=default["tau_refrac"] - 1,
            i_synin_gm_E=default["i_synin_gm_E"] - 1,
            i_synin_gm_I=default["i_synin_gm_I"] - 1,
            synapse_dac_bias=default["synapse_dac_bias"] - 1)

    def test_global_params(self):
        """
        Test correct assignment/masking in population views.
        """
        self.pop.set(synapse_dac_bias=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(synapse_dac_bias=400)

        self.pop.set(i_synin_gm_E=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(i_synin_gm_E=500)

        self.pop.set(i_synin_gm_I=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(i_synin_gm_I=500)


class TestCalibHXNeuronCoba(unittest.TestCase):

    def setUp(self):
        pynn.setup()
        self.pop = pynn.Population(3, pynn.cells.CalibHXNeuronCoba())
        self.pop.record(["spikes"])

    def tearDown(self):
        pynn.end()

    def test_empty(self):
        """
        Default calibrated neurons should be silent without excitation
        """
        pynn.run(10)
        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertEqual(len(spiketrain), 0)

    def test_external_stimulus(self):
        """
        Default calibrated neurons should fire with enough excitation
        """
        exc_stim_pop = pynn.Population(
            10, pynn.standardmodels.cells.SpikeSourcePoisson(
                rate=1e5, start=0, duration=10))

        pynn.Projection(
            exc_stim_pop, self.pop, pynn.AllToAllConnector(),
            synapse_type=pynn.standardmodels.synapses.StaticSynapse(weight=63),
            receptor_type="excitatory")
        pynn.run(10)

        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertGreater(len(spiketrain), 0)

    def test_leak_over_threshold(self):
        """
        Manually setting leak to max should lead to firing of the neuron.
        """
        pynn.preprocess()
        self.pop.actual_hwparams[0].leak.v_leak = 1022
        pynn.run(10)

        spiketrain = self.pop.get_data("spikes").segments[0].spiketrains[0]
        self.assertGreater(len(spiketrain), 0)

    def test_access_calib_result(self):
        """
        Calibration related parameters only available after run() or
        preprocess().
        """
        with self.assertRaises(AttributeError):
            _ = self.pop.calib_target
        with self.assertRaises(AttributeError):
            _ = self.pop.calib_hwparams
        with self.assertRaises(AttributeError):
            _ = self.pop.actual_hwparams

        pynn.preprocess()
        _ = self.pop.calib_target
        _ = self.pop.calib_hwparams
        _ = self.pop.actual_hwparams

    def test_popview(self):
        """
        Test correct assignment/masking in population views.
        """
        pynn.preprocess()
        # modifying second entry in the population view should change the third
        # neuron in the parent population
        self.pop[1:3].actual_hwparams[1].leak.v_leak = 1022
        self.assertEqual(self.pop.actual_hwparams[2].leak.v_leak, 1022)
        self.assertNotEqual(self.pop.actual_hwparams[0].leak.v_leak, 1022)
        self.assertNotEqual(self.pop.actual_hwparams[1].leak.v_leak, 1022)

    def test_non_default_params(self):
        """
        Test access to all parameters.
        """
        default = self.pop.celltype.default_parameters
        self.pop.set(
            v_rest=default["v_rest"] - 1,
            v_reset=default["v_reset"] - 1,
            v_thresh=default["v_thresh"] - 1,
            tau_m=default["tau_m"] - 1,
            tau_syn_E=default["tau_syn_E"] - 1,
            tau_syn_I=default["tau_syn_I"] - 1,
            cm=default["cm"] - 1,
            tau_refrac=default["tau_refrac"] - 1,
            e_rev_E=default["e_rev_E"] - 1,
            e_rev_I=default["e_rev_I"] - 1,
            i_synin_gm_E=default["i_synin_gm_E"] - 1,
            i_synin_gm_I=default["i_synin_gm_I"] - 1,
            synapse_dac_bias=default["synapse_dac_bias"] - 1)

    def test_global_params(self):
        """
        Test correct assignment/masking in population views.
        """
        self.pop.set(synapse_dac_bias=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(synapse_dac_bias=400)

        self.pop.set(i_synin_gm_E=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(i_synin_gm_E=500)

        self.pop.set(i_synin_gm_I=[400, 500, 500])
        with self.assertRaises(AttributeError):
            pynn.run(1)
        self.pop.set(i_synin_gm_I=500)


if __name__ == "__main__":
    unittest.main()
