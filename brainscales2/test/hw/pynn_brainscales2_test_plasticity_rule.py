#!/usr/bin/env python

import unittest
from pynn_brainscales.brainscales2.examples.plasticity_rule import main, \
    cell_params


class TestPlasticityRule(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main(cell_params)


if __name__ == "__main__":
    unittest.main()
