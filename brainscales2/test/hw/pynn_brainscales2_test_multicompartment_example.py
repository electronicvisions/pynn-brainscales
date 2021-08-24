#!/usr/bin/env python

import unittest
from pynn_brainscales.brainscales2.examples.multicompartment import main


class TestMulticompartment(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main()


if __name__ == "__main__":
    unittest.main()
