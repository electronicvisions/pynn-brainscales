#!/usr/bin/env python

import unittest
import pynn_brainscales.brainscales2 as pynn


class TestEmpty(unittest.TestCase):
    @staticmethod
    def test_empty():
        pynn.setup()
        pynn.run(0.2)
        pynn.end()


if __name__ == "__main__":
    unittest.main()
