import unittest
from examples.leak_over_threshold import main


class TestLeakOverThreshold(unittest.TestCase):
    def test_main(self):
        initial_values = {"threshold_v_threshold": 400,
                          "leak_reset_leak_v_leak": 1022,
                          "leak_reset_reset_v_reset": 50,
                          "leak_reset_leak_i_bias": 420,
                          "leak_reset_reset_i_bias": 950,
                          "leak_reset_leak_enable_division": True,
                          "threshold_enable": True,
                          "leak_reset_reset_enable_multiplication": True,
                          "membrane_capacitance_capacitance": 32,
                          "refractory_period_refractory_time": 100}

        # Simply tests if program runs and generates the plot
        self.assertIsNone(main(initial_values))


if __name__ == "__main__":
    unittest.main()
