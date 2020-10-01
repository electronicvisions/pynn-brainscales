#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


# TODO: Add test for other connectors (cf. feature #3646)
# TODO: Add test for random numbers (cf. feature #3647)

class TestProjection(unittest.TestCase):

    def setUp(self):
        self.pop1 = pynn.Population(1, pynn.cells.HXNeuron)
        self.pop2 = pynn.Population(1, pynn.cells.HXNeuron)
        self.pop3 = pynn.Population(2, pynn.cells.HXNeuron)
        self.pop4 = pynn.Population(3, pynn.cells.HXNeuron)

    def test_delay(self):
        proj = pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector())
        self.assertEqual(proj.get("delay", format="array"), [0])
        proj.set(delay=0)
        self.assertEqual(proj.get("delay", format="array"), [0])
        with self.assertRaises(ValueError):
            proj.set(delay=1)

    def test_weight(self):
        proj = pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector())
        self.assertEqual(proj.get("weight", format="array"), 0)
        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=32)
        proj = pynn.Projection(self.pop1, self.pop2, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        self.assertEqual(proj.get("weight", format="array"), 32)
        proj.set(weight=60.5)
        self.assertEqual(proj.get("weight", format="array"), 60)
        proj.set(weight=70)
        self.assertEqual(proj.get("weight", format="list"), [(0, 0, 70)])
        with self.assertRaises(ValueError):
            proj.set(weight=16129)
        with self.assertRaises(ValueError):
            proj.set(weight=-1)

        proj = pynn.Projection(self.pop3, self.pop4, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        connection_list = [(0, 0, 32), (1, 0, 32),
                           (0, 1, 32), (1, 1, 32),
                           (0, 2, 32), (1, 2, 32)]
        self.assertEqual(proj.get("weight", format="list"), connection_list)

        proj = pynn.Projection(self.pop4, self.pop3, pynn.AllToAllConnector(),
                               synapse_type=synapse)
        connection_list = [(0, 0, 32), (1, 0, 32), (2, 0, 32),
                           (0, 1, 32), (1, 1, 32), (2, 1, 32)]
        self.assertEqual(proj.get("weight", format="list"), connection_list)


if __name__ == '__main__':
    unittest.main()
