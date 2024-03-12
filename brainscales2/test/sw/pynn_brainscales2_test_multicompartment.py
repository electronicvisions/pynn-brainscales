#!/usr/bin/env python

import unittest
import copy
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.morphology import create_mc_neuron, \
    McCircuitParameters, Compartment, SharedLineConnection

from dlens_vx_v3 import halco, lola


class TestCompartment(unittest.TestCase):
    def test_correct_init(self):
        positions = [0, 1]
        label = 'my_label'

        # only positional arguments
        comp = Compartment(positions=positions, label=label)
        self.assertEqual(comp.positions, positions)
        self.assertEqual(comp.label, label)
        self.assertEqual(len(comp.connect_shared_line), 0)
        self.assertEqual(len(comp.connect_conductance), 0)

        # connect shared_line
        comp = Compartment(positions=positions, label=label,
                           connect_shared_line=positions[:1])
        self.assertEqual(len(comp.connect_shared_line), 1)

        # connect connect_conductance
        comp = Compartment(positions=positions, label=label,
                           connect_conductance=[(positions[1], 200)])
        self.assertEqual(len(comp.connect_conductance), 1)

        # parameters (only test construction)
        comp = Compartment(positions=positions, label=label, some_parameter=0,
                           some_other_param=[200, 300])

    def test_incorrect_init(self):
        positions = [0, 1]
        label = 'my_label'

        # positions non-unique
        with self.assertRaises(TypeError):
            Compartment(positions=positions + positions, label=label)

        # connect_shared_line
        with self.assertRaises(TypeError):
            # non-iterable
            Compartment(positions=positions,
                        label=label,
                        connect_shared_line=0)

        with self.assertRaises(TypeError):
            # non-unique
            Compartment(positions=positions,
                        label=label,
                        connect_shared_line=[positions[:1] + positions[:1]])

        with self.assertRaises(TypeError):
            # positon not in positions
            Compartment(positions=positions,
                        label=label,
                        connect_shared_line=[positions[-1] + 1])

        # connect_conductance
        with self.assertRaises(TypeError):
            # non-iterable
            Compartment(positions=positions,
                        label=label,
                        connect_conductance=0)

        with self.assertRaises(TypeError):
            # non-unique positions
            Compartment(positions=positions,
                        label=label,
                        connect_conductance=[(positions[1], 100),
                                             (positions[1], 200)])

        with self.assertRaises(TypeError):
            # positon not in positions
            Compartment(positions=positions,
                        label=label,
                        connect_conductance=[(positions[-1] + 1, 100)])

        with self.assertRaises(TypeError):
            # wrong shape (not list of tuples)
            Compartment(positions=positions,
                        label=label,
                        connect_conductance=(positions[:1], 100))

        with self.assertRaises(TypeError):
            # wrong shape (tuples with wrong size)
            Compartment(positions=positions,
                        label=label,
                        connect_conductance=(positions[:1], 100, 10))


class TestNeuronClass(unittest.TestCase):
    def test_correct_creation(self):
        # single compartment
        comp_0 = Compartment(positions=[0], label='label0')
        neuron_class = create_mc_neuron('McNeuron', compartments=[comp_0])
        self.assertEqual(len(neuron_class.compartments), 1)

        # multiple compartments
        comp_0 = Compartment(positions=[0], label='label0',
                             connect_shared_line=[0])
        comp_1 = Compartment(positions=[1], label='label1',
                             connect_conductance=[(1, 200)])
        neuron_class = create_mc_neuron(
            'McNeuron', compartments=[comp_0, comp_1],
            connections=[SharedLineConnection(start=0, stop=1, row=0)])
        self.assertEqual(len(neuron_class.compartments), 2)

    def test_incorrect_creation(self):
        comp_0 = Compartment(positions=[0], label='label0',
                             connect_shared_line=[0])
        comp_1 = Compartment(positions=[1], label='label1',
                             connect_conductance=[(1, 200)])
        connection = SharedLineConnection(start=0, stop=1, row=0)

        # Make sure base configuration works:
        create_mc_neuron('McNeuron', compartments=[comp_0, comp_1],
                         connections=[connection])

        # wrong compartments argument
        with self.assertRaises(TypeError):
            # non-iterable
            create_mc_neuron('McNeuron', compartments=comp_0)

        # wrong connections argument
        with self.assertRaises(TypeError):
            # non-iterable
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_1],
                             connections=connection)
        with self.assertRaises(TypeError):
            # wrong type
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_1],
                             connections=[(0, 1, 0)])

        # invalid morphology
        with self.assertRaises(RuntimeError):
            # compartments without connections
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_1])
        with self.assertRaises(RuntimeError):
            # direct connection between compartments
            comp_wrong = copy.deepcopy(comp_1)
            comp_wrong.connect_shared_line = [1]
            comp_wrong.connect_conductance = []
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_wrong],
                             connections=[connection])
        with self.assertRaises(RuntimeError):
            # same neuron circuit in two different compartments
            comp_wrong = copy.deepcopy(comp_1)
            comp_wrong.positions.extend(comp_0.positions)
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_wrong],
                             connections=[connection])
        with self.assertRaises(RuntimeError):
            # use connect_conductance and connect_shared_line at the same time
            comp_wrong = copy.deepcopy(comp_1)
            comp_wrong.connect_shared_line = [1]
            create_mc_neuron('McNeuron', compartments=[comp_0, comp_wrong],
                             connections=[connection])

    def test_retrival_of_labels(self):
        comp_0 = Compartment(positions=[0], label='my_label',
                             connect_shared_line=[0])
        comp_1 = Compartment(positions=[1], label='my_label',
                             connect_conductance=[(1, 200)])
        comp_2 = Compartment(positions=[2], label='my_other_label',
                             connect_conductance=[(2, 200)])
        connections = [SharedLineConnection(start=0, stop=2, row=0)]
        neuron_class = create_mc_neuron(
            'McNeuron', compartments=[comp_0, comp_1, comp_2],
            connections=connections)

        # retrieve all labels
        self.assertEqual(neuron_class.get_labels(),
                         ['my_label', 'my_label', 'my_other_label'])

        # retrieve label of single comaprtment
        self.assertEqual(
            neuron_class.get_label(halco.CompartmentOnLogicalNeuron(0)),
            'my_label')

        self.assertEqual(
            neuron_class.get_label(halco.CompartmentOnLogicalNeuron(2)),
            'my_other_label')

        # retrieve label of non-existent compartment
        with self.assertRaises(KeyError):
            neuron_class.get_label(halco.CompartmentOnLogicalNeuron(4))


class TestPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Create a simple multi-compartmental neuron model which can be used in
        the following tests.
        '''
        v_leak = [[1], [2, 3], [4, 4]]
        v_threshold = [[1], [2, 3], [4, 4]]
        comp_0 = Compartment(positions=[0], label='my_label',
                             connect_shared_line=[0],
                             leak_v_leak=v_leak[0],
                             threshold_v_threshold=v_threshold[0])
        comp_1 = Compartment(positions=[1, 2], label='my_label',
                             connect_conductance=[(1, 200)],
                             leak_v_leak=v_leak[1],
                             threshold_v_threshold=v_threshold[1])
        # assign scalar values
        comp_2 = Compartment(positions=[3, 4], label='my_label',
                             connect_conductance=[(3, 200)],
                             leak_v_leak=v_leak[2][0],
                             threshold_v_threshold=v_threshold[2][0])
        connections = [SharedLineConnection(start=0, stop=3, row=0)]
        cls.McNeuron = create_mc_neuron(
            'McNeuron', compartments=[comp_0, comp_1, comp_2],
            connections=connections)

        # values expected after initialization
        cls.v_leak = McCircuitParameters(v_leak)
        cls.v_threshold = McCircuitParameters(v_threshold)

        pynn.setup()

    def test_correct_default_values(self):
        pop = pynn.Population(1, self.McNeuron())

        # Get single value
        self.assertEqual(pop.get('leak_v_leak'), self.v_leak)

        # Get multiple values
        self.assertEqual(pop.get(['leak_v_leak', 'threshold_v_threshold']),
                         [self.v_leak, self.v_threshold])

        # Repeat tests with population_size > 1 (values should be collapsed
        # and the returns should be the same as above)
        pop = pynn.Population(5, self.McNeuron())
        # Get single value
        self.assertEqual(pop.get('leak_v_leak'), self.v_leak)

        # Get multiple values
        self.assertEqual(pop.get(['leak_v_leak', 'threshold_v_threshold']),
                         [self.v_leak, self.v_threshold])

        # Parameters provided to neuron class
        pop = pynn.Population(1, self.McNeuron(leak_v_leak=100))
        self.assertEqual(pop.get('leak_v_leak'), 100)

    def test_setting_values(self):
        pop = pynn.Population(1, self.McNeuron())
        new_value = copy.deepcopy(self.v_leak)
        new_value.value = self.v_leak.value + 1

        pop.set(leak_v_leak=new_value)
        self.assertEqual(pop.get('leak_v_leak'), new_value)

        # Repeat tests with population_size > 1
        pop = pynn.Population(2, self.McNeuron())

        pop.set(leak_v_leak=new_value)
        self.assertEqual(pop.get('leak_v_leak'), new_value)

        # set different values for different neurons
        new_values = np.array([self.v_leak, new_value])
        pop.set(leak_v_leak=new_values)
        self.assertTrue(np.all(pop.get('leak_v_leak') == new_values))

    @classmethod
    def tearDownClass(cls):
        pynn.end()


class TestCalibration(unittest.TestCase):
    def test_correct_default_values(self):
        v_leak = [[1], [9999]]
        v_threshold = [[1], [9999]]
        comp_0 = Compartment(positions=[0], label='my_label',
                             connect_shared_line=[0],
                             leak_v_leak=v_leak[0],
                             threshold_v_threshold=v_threshold[0])
        # leak_v_leak and threshold_v_threshold should be taken from
        # calibration
        comp_1 = Compartment(positions=[1], label='my_label',
                             connect_conductance=[(1, 200)])
        connections = [SharedLineConnection(start=0, stop=1, row=0)]
        McNeuron = create_mc_neuron(
            'McNeuron', compartments=[comp_0, comp_1], connections=connections)

        initial_config = lola.Chip()
        # set leak and threshold potential to AtomicNeuronOnDLS value for
        # easy checking
        for coord, neuron in enumerate(
                initial_config.neuron_block.atomic_neurons):  # pylint:disable=no-member
            neuron.leak.v_leak = coord
            neuron.threshold.v_threshold = coord

        pynn.setup(initial_config=initial_config)

        # values expected after initialization
        v_leak[1][0] = 1
        v_leak = McCircuitParameters(v_leak)
        v_threshold[1][0] = 1
        v_threshold = McCircuitParameters(v_threshold)

        pop = pynn.Population(1, McNeuron())

        # Get single value
        self.assertEqual(pop.get('leak_v_leak'), v_leak)

        # Get multiple values
        self.assertEqual(pop.get(['leak_v_leak', 'threshold_v_threshold']),
                         [v_leak, v_threshold])

        # Parameters provided to neuron class
        pop = pynn.Population(1, McNeuron(leak_v_leak=100))
        self.assertEqual(pop.get('leak_v_leak'), 100)

        pynn.end()


class TestExecution(unittest.TestCase):
    @staticmethod
    def test_single_compartment():
        comp_0 = Compartment(positions=[0], label='my_label')
        McNeuron = create_mc_neuron('McNeuron', compartments=[comp_0])

        pynn.setup()
        pynn.Population(1, McNeuron())
        pynn.run(None, pynn.RunCommand.PREPARE)
        pynn.end()

        # multiple circuits
        comp_0 = Compartment(positions=[0, 1], label='my_label')
        McNeuron = create_mc_neuron('McNeuron', compartments=[comp_0])

        pynn.setup()
        pynn.Population(1, McNeuron())
        pynn.run(None, pynn.RunCommand.PREPARE)
        pynn.end()

    @staticmethod
    def test_multiple_compartments():
        comp_0 = Compartment(positions=[0], label='my_label',
                             connect_shared_line=[0])
        comp_1 = Compartment(positions=[1], label='my_label',
                             connect_conductance=[(1, 200)])
        connections = [SharedLineConnection(start=0, stop=1, row=0)]
        McNeuron = create_mc_neuron(
            'McNeuron', compartments=[comp_0, comp_1], connections=connections)

        pynn.setup()
        pynn.Population(1, McNeuron())
        pynn.run(None, pynn.RunCommand.PREPARE)
        pynn.end()


if __name__ == '__main__':
    unittest.main()
