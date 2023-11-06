#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


class TestEmpty(unittest.TestCase):
    @staticmethod
    def test_empty():
        pynn.setup()
        pynn.run(0.2)
        pynn.end()

    def test_several_runs(self):
        pynn.setup()
        pynn.run(0.2)

        with self.assertRaises(RuntimeError):
            pynn.run(0.2)
        pynn.end()


if __name__ == "__main__":
    unittest.main()
