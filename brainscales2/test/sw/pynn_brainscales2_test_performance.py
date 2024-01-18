#!/usr/bin/env python

import time
import unittest
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v3 import logger


logger.set_loglevel(logger.get('TestPerformance'), logger.LogLevel.INFO)


# This only executes and prints performance, no assertions are made
class TestPerformance(unittest.TestCase):
    def setUp(self):
        pynn.setup()

    def tearDown(self):
        pynn.end()

    @classmethod
    def test_many_small_projections(cls):
        log = logger.get("TestPerformance.test_many_small_projections")
        begin = time.time()
        pops = []
        for i in range(256):
            pop = pynn.Population(1, pynn.cells.HXNeuron())
            # wrap as View because using it in projections is faster
            pops.append(pynn.PopulationView(pop, np.array(range(len(pop)))))

        projs = []
        for i in range(256):
            for j in range(256):
                projs.append(pynn.Projection(pops[i], pops[j],
                                             pynn.AllToAllConnector()))
        log.INFO(f"network construction {time.time() - begin}s")

        begin = time.time()
        pynn.add(1) #does not perform hardware run. just preprocessing
        log.INFO(f"first run {time.time() - begin}s")

        begin = time.time()
        projs[-1].set(weight=10)
        log.INFO(f"network modification (weights) {time.time() - begin}s")

        for i in range(10):
            begin = time.time()
            pynn.add(1)
            log.INFO(f"run {i} after modification {time.time() - begin}s")

    @classmethod
    def test_few_large_projections(cls):
        log = logger.get("TestPerformance.test_few_large_projections")
        begin = time.time()
        pop = pynn.Population(256, pynn.cells.HXNeuron())
        pops = []
        # wrap as View because using it in projections is faster
        pops.append(pynn.PopulationView(pop, np.array(range(len(pop)))))

        projs = []
        projs.append(pynn.Projection(pops[0], pops[0],
                                     pynn.AllToAllConnector()))
        log.INFO(f"network construction {time.time() - begin}s")

        begin = time.time()
        pynn.add(1)
        log.INFO(f"first run {time.time() - begin}s")

        begin = time.time()
        projs[-1].set(weight=10)
        log.INFO(f"network modification (weights) {time.time() - begin}s")

        for i in range(10):
            begin = time.time()
            pynn.add(1)
            log.INFO(f"run {i} after modification {time.time() - begin}s")


if __name__ == '__main__':
    unittest.main()
