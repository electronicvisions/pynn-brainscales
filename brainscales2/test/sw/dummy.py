import unittest

from pynn_brainscales.brainscales2 import add


class TestAdder(unittest.TestCase):
    def test_commutative(self):
        self.assertEqual(add(3, 4), add(4, 3))
