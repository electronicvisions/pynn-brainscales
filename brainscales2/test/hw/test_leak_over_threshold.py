import unittest
from examples.leak_over_threshold import main


class TestLeakOverThreshold(unittest.TestCase):
    def test_main(self):
        cell_params = {"threshold_v_threshold": 400,
                       "leak_v_leak": 1022,
                       "leak_i_bias": 420,
                       "leak_enable_division": True,
                       "reset_v_reset": 50,
                       "reset_i_bias": 950,
                       "reset_enable_multiplication": True,
                       "threshold_enable": True,
                       "membrane_capacitance_capacitance": 32,
                       "refractory_period_refractory_time": 100}

        # Simply tests if program runs and generates the plot
        self.assertIsNone(main(cell_params))


if __name__ == "__main__":
    unittest.main()
