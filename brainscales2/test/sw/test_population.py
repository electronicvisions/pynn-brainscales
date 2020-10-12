#!/usr/bin/env python

import unittest
import inspect
import numpy as np
from dlens_vx_v2 import lola
import pynn_brainscales.brainscales2 as pynn


# To be added: PopulationView Test
_simulator = pynn.simulator


class TestAPopulation(unittest.TestCase):

    def setUp(self):
        self.pop1 = pynn.Population(1, pynn.cells.HXNeuron)
        self.pop2 = pynn.Population(5, pynn.cells.HXNeuron, initial_values={
            "threshold_v_threshold": 200})
        self.pop3 = pynn.Population(3, pynn.cells.HXNeuron, initial_values={
            "threshold_v_threshold": 300,
            "leak_i_bias": 100,
            "exponential_enable": True})

    def test_size(self):
        self.assertEqual(self.pop1.size, 1)
        self.assertEqual(self.pop2.size, 5)
        self.assertEqual(self.pop3.size, 3)

    def test_celltype(self):
        self.assertIsInstance(self.pop1.celltype, pynn.cells.HXNeuron)
        self.assertIsInstance(self.pop2.celltype, pynn.cells.HXNeuron)
        self.assertIsInstance(self.pop3.celltype, pynn.cells.HXNeuron)

    # works if this test is called first, since the Populations are newly
    # instanced for each test
    def test_cellids(self):
        np.testing.assert_array_equal(self.pop1.all_cells,
                                      np.array(range(0, self.pop1.size),
                                               dtype=_simulator.ID))
        np.testing.assert_array_equal(self.pop2.all_cells,
                                      np.array(range(self.pop1.size,
                                                     self.pop1.size
                                                     + self.pop2.size),
                                               dtype=_simulator.ID))
        np.testing.assert_array_equal(self.pop3.all_cells,
                                      np.array(range(self.pop1.size
                                                     + self.pop2.size,
                                                     self.pop1.size
                                                     + self.pop2.size
                                                     + self.pop3.size),
                                               dtype=_simulator.ID))

    def test_inital_values(self):
        self.assertEqual(self.pop2.get("threshold_v_threshold"), 200)
        self.assertEqual(self.pop3.get(["threshold_v_threshold",
                                        "leak_i_bias"]),
                         [300, 100])
        self.assertTrue(self.pop3.get("exponential_enable"))

    def test_not_configurable(self):
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "event_routing_analog_output": 0})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "event_routing_enable_digital": False})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "leak_reset_i_bias_source_follower": 0})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "readout_enable_amplifier": True})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "readout_source": "something"})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "readout_enable_buffered_access": True})
        with self.assertRaises(ValueError):
            pynn.Population(1, pynn.cells.HXNeuron, initial_values={
                "readout_i_bias": 1000})


class TestLolaNeuronConstruction(unittest.TestCase):

    def setUp(self):
        self.pop = pynn.Population(2, pynn.cells.HXNeuron, initial_values={
            "threshold_v_threshold": 200, "threshold_enable": 200})
        self.neurons = [None] * self.pop.size

        for i in range(self.pop.size):
            neuron = pynn.cells.HXNeuron.lola_from_dict(
                self.pop.initial_values)
            self.neurons[i] = neuron

    def test_equal(self):
        self.assertEqual(self.neurons[0], self.neurons[1])

    def test_initial_values(self):

        # get initial values of the HXNeuron
        params = pynn.cells.HXNeuron.ATOMIC_NEURON_MEMBERS

        for param in params:
            pop_mem = getattr(self.neurons[0], param)
            for attr, _ in inspect.getmembers(pop_mem):
                pop_member = getattr(pop_mem, attr)
                if attr.startswith("_") or not attr.islower() \
                    or isinstance(pop_member,
                                  lola.AtomicNeuron.EventRouting):
                    continue

                key = param + "_" + attr
                value = getattr(pop_mem, attr)
                # check values
                if param == "threshold":
                    if attr == "enable":
                        self.assertTrue(value)
                    elif attr == "v_threshold":
                        self.assertEqual(value, 200)
                    else:
                        self.assertEqual(value, pynn.cells.HXNeuron.
                                         default_initial_values[key])
                else:
                    self.assertEqual(value, pynn.cells.HXNeuron.
                                     default_initial_values[key])


if __name__ == '__main__':
    unittest.main()
