import unittest
from examples.external_input import main, init_values


class TestExternalInput(unittest.TestCase):
    def test_main(self):
        # Simply tests if program runs
        main(init_values)


if __name__ == "__main__":
    unittest.main()
