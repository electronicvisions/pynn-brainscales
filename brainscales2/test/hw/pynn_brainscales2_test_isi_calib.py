#!/usr/bin/env python

import unittest
from examples import isi_calib


class TestISICalib(unittest.TestCase):
    def test_calibrate_isi(self):
        target_isi = 0.01
        calibrated_isi, _ = isi_calib.calibrate_isi(target_isi)
        uncertainty = 0.001
        self.assertLessEqual(abs(target_isi - calibrated_isi), uncertainty)


if __name__ == "__main__":
    unittest.main()
