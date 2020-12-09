#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


# TODO: Add test for other connectors (cf. feature #3646)
# TODO: Add test for random numbers (cf. feature #3647)

class TestProjection(unittest.TestCase):

    def setUp(self):
        pynn.setup()
        self.pop1 = pynn.Population(1, pynn.cells.HXNeuron())
        self.pop2 = pynn.Population(1, pynn.cells.HXNeuron())
        self.pop3 = pynn.Population(2, pynn.cells.HXNeuron())
        self.pop4 = pynn.Population(3, pynn.cells.HXNeuron())

    def tearDown(self):
        pynn.end()

    def test_identical_connection(self):
        pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector())
        pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector())
        pynn.run(None)

    def test_delay(self):
        proj = pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector())
        self.assertEqual(proj.get("delay", format="array"), [0])
        proj.set(delay=0)
        self.assertEqual(proj.get("delay", format="array"), [0])
        with self.assertRaises(ValueError):
            proj.set(delay=1)
        pynn.run(None)

    def test_weight(self):
        proj = pynn.Projection(self.pop1, self.pop1, pynn.AllToAllConnector())
        self.assertEqual(proj.get("weight", format="array"), 0)
        pynn.run(None)
        pynn.simulator.state.projections = []
        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=32)
        proj = pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        self.assertEqual(proj.get("weight", format="array"), 32)
        proj.set(weight=42.5)
        self.assertEqual(proj.get("weight", format="array"), 42)
        proj.set(weight=43)
        self.assertEqual(proj.get("weight", format="list"), [(0, 0, 43)])
        with self.assertRaises(ValueError):
            proj.set(weight=64)
        with self.assertRaises(pynn.errors.ConnectionError):
            proj.set(weight=-1)

        proj = pynn.Projection(self.pop3, self.pop4, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        connection_list = [(0, 0, 32), (0, 1, 32),
                           (0, 2, 32), (1, 0, 32),
                           (1, 1, 32), (1, 2, 32)]
        self.assertEqual(proj.get("weight", format="list"), connection_list)

        proj = pynn.Projection(self.pop4, self.pop3, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        connection_list = [(0, 0, 32), (0, 1, 32), (1, 0, 32),
                           (1, 1, 32), (2, 0, 32), (2, 1, 32)]
        self.assertEqual(proj.get("weight", format="list"), connection_list)
        pynn.run(None)

    def test_weight_sign(self):
        synapse_exc = pynn.standardmodels.synapses.StaticSynapse(weight=32)
        synapse_inh = pynn.standardmodels.synapses.StaticSynapse(weight=-32)
        proj_exc = pynn.Projection(
            self.pop1, self.pop1, pynn.AllToAllConnector(),
            receptor_type="excitatory", synapse_type=synapse_exc)
        proj_inh = pynn.Projection(
            self.pop1, self.pop1, pynn.AllToAllConnector(),
            receptor_type="inhibitory", synapse_type=synapse_inh)

        # wrong sign on construction
        with self.assertRaises(pynn.errors.ConnectionError):
            pynn.Projection(
                self.pop1, self.pop1, pynn.AllToAllConnector(),
                receptor_type="excitatory", synapse_type=synapse_inh)
        with self.assertRaises(pynn.errors.ConnectionError):
            pynn.Projection(
                self.pop1, self.pop1, pynn.AllToAllConnector(),
                receptor_type="inhibitory", synapse_type=synapse_exc)

        # wrong sign on set()
        with self.assertRaises(pynn.errors.ConnectionError):
            proj_exc.set(weight=-32)
        with self.assertRaises(pynn.errors.ConnectionError):
            proj_inh.set(weight=32)

        pynn.run(None)

    def test_set_one_to_one(self):
        proj = pynn.Projection(self.pop3, self.pop3, pynn.OneToOneConnector())
        proj.set(weight=32)
        connection_list = [(0, 0, 32), (1, 1, 32)]
        self.assertEqual(proj.get("weight", format="list"), connection_list)
        pynn.run(None)
        pynn.simulator.state.projections = []

    def test_projection_view(self):
        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=32)
        proj = pynn.Projection(pynn.PopulationView(self.pop3, [1]),
                               pynn.PopulationView(self.pop4, [2]),
                               pynn.AllToAllConnector(),
                               synapse_type=synapse)
        self.assertEqual(proj.get("weight", format="list"), [(0, 0, 32)])


if __name__ == '__main__':
    unittest.main()
