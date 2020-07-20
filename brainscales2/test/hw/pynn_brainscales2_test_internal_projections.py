#!/usr/bin/env python

import unittest
from examples.internal_projections import main


class TestTryProjections(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main()


if __name__ == "__main__":
    unittest.main()
