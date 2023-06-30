#!/usr/bin/env python

import unittest
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales import errors, parameters
from pynn_brainscales.brainscales2.morphology import create_mc_neuron, \
    Compartment, SharedLineConnection


# To be added: PopulationView Test
_simulator = pynn.simulator


class TestAPopulation(unittest.TestCase):

    @staticmethod
    def _define_mc_neuron_class():
        comp_0 = Compartment(positions=[0], label='label0',
                             connect_shared_line=[0])
        comp_1 = Compartment(positions=[1], label='label1',
                             connect_conductance=[(1, 200)])
        return create_mc_neuron(
            'McNeuron',
            compartments=[comp_0, comp_1],
            connections=[SharedLineConnection(start=0, stop=1, row=0)])

    def setUp(self):
        pynn.setup()
        self.hxpop1 = pynn.Population(1, pynn.cells.HXNeuron())
        self.hxpop2 = pynn.Population(
            5, pynn.cells.HXNeuron(threshold_v_threshold=200))
        self.hxpop3 = pynn.Population(
            3, pynn.cells.HXNeuron(
                threshold_v_threshold=300,
                leak_i_bias=100,
                exponential_enable=True))
        self.hxpop4 = pynn.Population(
            2, pynn.cells.HXNeuron(leak_i_bias=[100, 150]))

        McNeuron = self._define_mc_neuron_class()
        self.mcpop1 = pynn.Population(1, McNeuron())

        # test old API support
        self.hxpop5 = pynn.Population(
            2, pynn.cells.HXNeuron, cellparams={'leak_i_bias': [100, 200]})
        self.sapop1 = pynn.Population(
            1, pynn.cells.SpikeSourceArray(spike_times=[0, 1, 2]))
        self.sapop2 = pynn.Population(
            2, pynn.cells.SpikeSourceArray(spike_times=[[0, 1, 2], [3, 4, 5]]))

    def tearDown(self):
        pynn.end()

    def test_size(self):
        self.assertEqual(self.hxpop1.size, 1)
        self.assertEqual(self.hxpop2.size, 5)
        self.assertEqual(self.hxpop3.size, 3)
        self.assertEqual(self.sapop1.size, 1)
        self.assertEqual(self.sapop2.size, 2)

    def test_celltype(self):
        self.assertIsInstance(self.hxpop1.celltype, pynn.cells.HXNeuron)
        self.assertIsInstance(self.hxpop2.celltype, pynn.cells.HXNeuron)
        self.assertIsInstance(self.hxpop3.celltype, pynn.cells.HXNeuron)
        self.assertIsInstance(
            self.sapop1.celltype, pynn.cells.SpikeSourceArray)
        self.assertIsInstance(
            self.sapop2.celltype, pynn.cells.SpikeSourceArray)

    # works if this test is called first, since the Populations are newly
    # instanced for each test
    def test_cellids(self):
        np.testing.assert_array_equal(self.hxpop1.all_cells,
                                      np.array(range(0, self.hxpop1.size),
                                               dtype=_simulator.ID))
        np.testing.assert_array_equal(self.hxpop2.all_cells,
                                      np.array(range(self.hxpop1.size,
                                                     self.hxpop1.size
                                                     + self.hxpop2.size),
                                               dtype=_simulator.ID))
        np.testing.assert_array_equal(self.hxpop3.all_cells,
                                      np.array(range(self.hxpop1.size
                                                     + self.hxpop2.size,
                                                     self.hxpop1.size
                                                     + self.hxpop2.size
                                                     + self.hxpop3.size),
                                               dtype=_simulator.ID))

    def test_accessors(self):
        self.assertEqual(self.hxpop2.get("threshold_v_threshold"), 200)
        self.assertEqual(
            self.hxpop3.get(["threshold_v_threshold", "leak_i_bias"]),
            [300, 100])
        self.assertTrue(self.hxpop3.get("exponential_enable"), True)
        self.assertTrue(np.array_equal(
            self.hxpop4.get("leak_i_bias"), [100, 150]))
        self.assertEqual(self.hxpop4[0:1].get("leak_i_bias"), 100)
        self.assertEqual(self.hxpop4[0].leak_i_bias, 100)
        self.assertEqual(self.hxpop4[1:2].get("leak_i_bias"), 150)
        self.assertEqual(self.hxpop4[1].leak_i_bias, 150)
        self.assertTrue(np.array_equal(
            self.hxpop5.get("leak_i_bias"), [100, 200]))
        self.assertEqual(
            self.sapop1.get("spike_times"), parameters.Sequence([0, 1, 2]))
        self.assertEqual(
            self.sapop1[0].spike_times, parameters.Sequence([0, 1, 2]))
        self.assertTrue(np.array_equal(
            self.sapop2.get("spike_times"),
            [parameters.ArrayParameter([0, 1, 2]),
             parameters.ArrayParameter([3, 4, 5])]))
        self.assertEqual(
            self.sapop2[0].spike_times, parameters.Sequence([0, 1, 2]))
        self.assertEqual(
            self.sapop2[1].spike_times, parameters.Sequence([3, 4, 5]))
        mypop = pynn.Population(3, pynn.cells.HXNeuron())
        mypop.set(leak_i_bias=400)
        mypop[0:2].set(leak_i_bias=150)
        self.assertTrue(
            np.array_equal(mypop.get("leak_i_bias"), [150, 150, 400]))
        mypop.set(leak_i_bias=[100, 200, 300])
        self.assertTrue(
            np.array_equal(mypop.get("leak_i_bias"), [100, 200, 300]))
        mypop[1].leak_i_bias = 350
        self.assertTrue(
            np.array_equal(mypop.get("leak_i_bias"), [100, 350, 300]))
        mypop.set(exponential_enable=True)
        mypop[1, 2].set(exponential_enable=False)
        self.assertTrue(np.array_equal(
            mypop.get("exponential_enable"),
            [True, False, False]))
        mypop.set(leak_i_bias=123)
        mypop[0, 2].set(exponential_enable=True, leak_i_bias=100)
        self.assertTrue(np.array_equal(
            mypop.get(["leak_i_bias", "exponential_enable"]),
            [[100, 123, 100], [True, False, True]]))

    def test_not_configurable(self):
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(event_routing_analog_output=0))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(event_routing_enable_digital=False))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(leak_reset_i_bias_source_follower=0))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(readout_enable_amplifier=True))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(readout_source="something"))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(readout_enable_buffered_access=True))
        with self.assertRaises(errors.NonExistentParameterError):
            pynn.Population(
                1, pynn.cells.HXNeuron(readout_i_bias=1000))

    def test_record(self):
        # test all valid madc types
        for value in ["v", "exc_synin", "inh_synin", "adaptation"]:
            self.hxpop1.record(value)
            self.hxpop1.record(None)
        self.hxpop1.record("spikes")
        self.hxpop1.record(None)

        # multi population recording
        self.hxpop1.record(["spikes", "v"])
        self.hxpop2.record("spikes")
        self.hxpop1.record(None)
        self.hxpop2.record(None)

        # MADC record for pop view of size one should work
        self.hxpop2[0:1].record("v")
        self.hxpop2.record(None)

        # only one MADC readout per neuron allowed
        self.hxpop1.record("v")
        with self.assertRaises(ValueError):
            self.hxpop1.record("inh_synin")
        self.hxpop1.record(None)
        with self.assertRaises(ValueError):
            self.hxpop1.record(["v", "inh_synin"])

        with self.assertRaises(errors.RecordingError):
            self.hxpop1.record("undefined_variable")

        # test locations argument
        for location in ['label0', 'label1']:
            self.mcpop1.record('v', locations=[location])
            self.mcpop1.record(None)
            self.mcpop1[0:1].record('v', locations=[location])
            self.mcpop1[0:1].record(None)

        # non-existent label
        with self.assertRaises(ValueError):
            self.mcpop1.record('v', locations=['non_existent_label'])
        self.mcpop1.record(None)


class TestLolaNeuronConstruction(unittest.TestCase):

    def setUp(self):
        pynn.setup()
        self.pop = pynn.Population(2, pynn.cells.HXNeuron(
            threshold_v_threshold=200, threshold_enable=200))
        self.neurons = [None] * self.pop.size

        for idx, item in enumerate(self.pop.celltype.parameter_space):
            neuron = pynn.cells.HXNeuron.create_hw_entity(item)
            self.neurons[idx] = neuron

    def tearDown(self):
        pynn.end()

    def test_equal(self):
        self.assertEqual(self.neurons[0], self.neurons[1])


if __name__ == '__main__':
    unittest.main()
