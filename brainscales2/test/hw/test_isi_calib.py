import unittest
from examples import isi_calib


class TestISICalib(unittest.TestCase):
    def test_calibrate_isi(self):
        target_isi = 0.04
        calibrated_isi, _ = isi_calib.calibrate_isi(0.04)
        uncertainty = 0.001
        self.assertLessEqual(abs(target_isi - calibrated_isi), uncertainty)


if __name__ == "__main__":
    unittest.main()
