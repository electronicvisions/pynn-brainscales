#!/usr/bin/env python3

import unittest
import pynn_brainscales.brainscales2 as pynn

from dlens_vx_v3 import hxcomm


class TestPassInConnection(unittest.TestCase):
    @staticmethod
    def test_pass_in_connection():
        with hxcomm.ManagedConnection() as connection:
            for _ in range(5):
                pynn.setup(connection=connection)
                pynn.Population(512, pynn.cells.HXNeuron())
                pynn.run(0.2)
                pynn.end()


if __name__ == "__main__":
    unittest.main()
