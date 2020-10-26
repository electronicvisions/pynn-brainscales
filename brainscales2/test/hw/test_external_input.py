import unittest
from examples.external_input import main, init_values


class TestExternalInput(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main(init_values)


if __name__ == "__main__":
    unittest.main()
