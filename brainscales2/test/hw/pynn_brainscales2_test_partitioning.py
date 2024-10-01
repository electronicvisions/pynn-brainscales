#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.examples import partitioned_network


pynn.logger.default_config(level=pynn.logger.LogLevel.TRACE)


class TestPartitioning(unittest.TestCase):
    def setUp(self):
        pynn.setup(enable_neuron_bypass=True)

    def tearDown(self):
        pynn.end()

    @classmethod
    def test_v_recording(cls):
        pop = pynn.Population(100, pynn.cells.HXNeuron())
        pop.record(["v"])

        pynn.run(1)


class TestPartitioningExample(unittest.TestCase):
    @classmethod
    def test_populations(cls):
        # for now test that it can be executed
        partitioned_network.main()


if __name__ == '__main__':
    unittest.main()
