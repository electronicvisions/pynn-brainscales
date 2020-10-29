import unittest
from examples.external_input import main, cell_params


class TestExternalInput(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main(cell_params)


if __name__ == "__main__":
    unittest.main()
