import unittest
from examples.external_input import main


class TestExternalInput(unittest.TestCase):
    def test_spikes(self):
        spike_number, _, _ = main()
        spike_expectation = 10
        spike_range = 1
        spike_difference = abs(spike_expectation - spike_number)
        self.assertLessEqual(spike_difference, spike_range)


if __name__ == "__main__":
    unittest.main()
