import unittest
from examples.internal_projections import main


class TestTryProjections(unittest.TestCase):
    def test_main(self):
        # Simply tests if program runs
        main()


if __name__ == "__main__":
    unittest.main()
