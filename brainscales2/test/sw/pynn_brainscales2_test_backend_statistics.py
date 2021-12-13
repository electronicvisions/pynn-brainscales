#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


class TestBackendStatistics(unittest.TestCase):
    def test(self):
        # no active simulator
        with self.assertRaises(RuntimeError):
            pynn.get_backend_statistics()

        pynn.setup()
        pop1 = pynn.Population(1, pynn.cells.HXNeuron())
        pop2 = pynn.Population(1, pynn.cells.HXNeuron())
        pynn.Projection(pop1, pop2, pynn.AllToAllConnector())

        # no mapping and routing execution
        with self.assertRaises(RuntimeError):
            pynn.get_backend_statistics()

        pynn.run(None)
        string = str(pynn.get_backend_statistics())
        self.assertTrue("NetworkGraphStatistics" in string)


if __name__ == '__main__':
    unittest.main()
