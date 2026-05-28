#!/usr/bin/env python

"""
Test multi-compartment builder.

The builder itself is tested in grenade. Here we test populations
and projections which include neurons created with the builder.

Since a calcibration is executed in `pynn.run(run, pynn.RunComman.PREPARE)`,
this has to be a hardware test (for now).
"""

import unittest
import copy
import numpy as np

from pynn_brainscales.brainscales2.morphology.builder import MorphologyBuilder

from pygrenade_vx.network.abstract.multicompartment import mechanisms

from pygrenade_vx.network.abstract.multicompartment.tree import Connection
from pygrenade_vx.network.abstract.multicompartment.compartment_builder \
    import CompartmentBuilder

import pynn_brainscales.brainscales2 as pynn


def create_chain(length: int):
    """
    Create a chain of compartments.

    The first compartment is active and features a fire mechanisms "fire".
    All other compartments are passive. All compartments have the following
    mechanisms:
      - cap: capacitance
      - syn1: current-based synapse
      - syn2: current-based synapse
      - leak: leak
    """

    def add_passive_mechanisms(c_builder):
        c_builder.add(mechanisms.MembraneCapacitance(), label='cap')
        c_builder.add(mechanisms.CurrentBasedSynapse(), label='syn1')
        c_builder.add(mechanisms.CurrentBasedSynapse(), label='syn2')
        c_builder.add(mechanisms.Leak(), label='leak')

    # construct compartment
    comp_builder = CompartmentBuilder()
    add_passive_mechanisms(comp_builder)
    passive_comp = comp_builder.done("PassiveCompartment")

    comp_builder = CompartmentBuilder()
    add_passive_mechanisms(comp_builder)
    comp_builder.add(mechanisms.Fire(), label='fire')
    active_comp = comp_builder.done("ActiveCompartment")

    # Construct neuron
    builder = MorphologyBuilder()
    nodes = []
    nodes.append(builder.add_compartment(active_comp(),
                                         label="C0"))
    for n_comp in range(1, length):
        nodes.append(
            builder.add_compartment(passive_comp(), label=f"C{n_comp}"))

    connections = [Connection(first, second, 10e-6) for first, second in
                   zip(nodes[:-1], nodes[1:])]

    builder.connect(connections)
    return builder.done("MyNeuron")


class TestPopulation(unittest.TestCase):
    """
    Test populations.
    """

    @classmethod
    def setUpClass(cls):
        pynn.setup()

    def test_correct_default_values(self):
        pop = pynn.Population(1, create_chain(3)(
            **{"C0.fire.v_threshold": 300, "C1.cap.capacitance": 1e-12}))

        # Get single value
        self.assertEqual(pop.get('C0.fire.v_threshold'), 300)

        # Get multiple values
        self.assertEqual(pop.get(['C0.fire.v_threshold',
                                  'C1.cap.capacitance']),
                         [300, 1e-12])

        # Repeat tests with population_size > 1 (values should be collapsed
        # and the returns should be the same as above)
        pop = pynn.Population(5, create_chain(3)(
            **{"C0.fire.v_threshold": 300, "C1.cap.capacitance": 1e-12}))

        self.assertEqual(pop.get('C0.fire.v_threshold'), 300)
        self.assertEqual(pop.get(['C0.fire.v_threshold',
                                  'C1.cap.capacitance']),
                         [300, 1e-12])

    def test_setting_values(self):
        pop = pynn.Population(1, create_chain(3)())

        threshold = copy.deepcopy(pop.get('C0.fire.v_threshold'))
        new_value = threshold + 1
        pop.set(**{'C0.fire.v_threshold': new_value})

        self.assertEqual(pop.get('C0.fire.v_threshold'), new_value)

        # Repeat tests with population_size > 1
        pop = pynn.Population(2, create_chain(3)())

        pop.set(**{'C0.fire.v_threshold': new_value})

        self.assertEqual(pop.get('C0.fire.v_threshold'), new_value)

        # set different values for different neurons
        new_values = np.array([threshold - 1, new_value])
        pop.set(**{'C0.fire.v_threshold': new_values})
        self.assertTrue(np.all(pop.get('C0.fire.v_threshold') == new_values))

    @classmethod
    def tearDownClass(cls):
        pynn.end()


class TestProjections(unittest.TestCase):
    """
    Test projections.
    """

    @classmethod
    def setUpClass(cls):
        pynn.setup()

    def test_post(self):
        """
        Created neuron is post-synaptic neuron.
        """
        pop = pynn.Population(1, create_chain(3)())
        in_pop = pynn.Population(1, pynn.cells.SpikeSourceArray())
        pynn.Projection(
            in_pop, pop,
            pynn.AllToAllConnector(location_selector='C0'),
            receptor_type="syn1")
        pynn.run(None, pynn.RunCommand.PREPARE)

        pynn.Projection(
            in_pop, pop,
            pynn.AllToAllConnector(location_selector='C0'),
            receptor_type="syn2")
        pynn.run(None, pynn.RunCommand.PREPARE)

        with self.assertRaises(ValueError):
            pynn.Projection(
                in_pop, pop,
                pynn.AllToAllConnector(location_selector='not_specified'),
                receptor_type="syn2")

        with self.assertRaises(pynn.errors.ConnectionError):
            pynn.Projection(
                in_pop, pop,
                pynn.AllToAllConnector(location_selector='C0'),
                receptor_type="not_specified")

    def test_pre(self):
        """
        Created neuron is pre-synaptic neuron.
        """
        pop = pynn.Population(1, create_chain(3)())
        target_pop = pynn.Population(1, pynn.cells.CalibHXNeuronCuba())
        pynn.Projection(
            pop, target_pop,
            pynn.AllToAllConnector(source_location_selector='C0'))
        pynn.run(None, pynn.RunCommand.PREPARE)

    def test_pre_and_post(self):
        """
        Created neuron is pre-synaptic neuron.
        """
        pop = pynn.Population(1, create_chain(3)())
        target_pop = pynn.Population(1, create_chain(3)())
        pynn.Projection(
            pop, target_pop,
            pynn.AllToAllConnector(source_location_selector='C0',
                                   location_selector='C2'),
            receptor_type="syn1")
        pynn.run(None, pynn.RunCommand.PREPARE)

    @classmethod
    def tearDownClass(cls):
        pynn.end()


class TestRecording(unittest.TestCase):
    """
    Test that recording can be specified.
    """

    @classmethod
    def setUpClass(cls):
        pynn.setup()

    def test_spike_recording(self):
        """
        Created neuron is post-synaptic neuron.
        """
        pop = pynn.Population(1, create_chain(3)())
        pop.record("fire")
        pynn.run(None, pynn.RunCommand.PREPARE)

    def test_analog_recording(self):
        """
        Created neuron is post-synaptic neuron.
        """
        pop = pynn.Population(1, create_chain(3)())
        pop.record("cap")
        pynn.run(None, pynn.RunCommand.PREPARE)

    @classmethod
    def tearDownClass(cls):
        pynn.end()


if __name__ == '__main__':
    unittest.main()
