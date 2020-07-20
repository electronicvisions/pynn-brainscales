import unittest
from examples.internal_projections import main


class TestTryProjections(unittest.TestCase):
    def test_spikes(self):
        spike_number, _, _ = main()
        spike_expectation = 20
        spike_range = 2
        spike_difference = abs(spike_expectation - spike_number)
        self.assertLessEqual(spike_difference, spike_range)


if __name__ == "__main__":
    unittest.main()
