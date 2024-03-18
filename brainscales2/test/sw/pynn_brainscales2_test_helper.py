#!/usr/bin/env python

import os
import unittest
from unittest import mock
from pathlib import Path
import pynn_brainscales.brainscales2 as pynn


class TestHelper(unittest.TestCase):

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_nightly_calib_path(self):
        expected_path = "/wang/data/calibration/hicann-dls-sr-hx/zeromock/" \
            "stable/latest/spiking_cocolist.pbin"
        tested_path = pynn.helper.nightly_calib_path()
        self.assertEqual(expected_path, str(tested_path))

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_nightly_chip_extraction(self):
        chip = pynn.helper.chip_from_nightly()
        self.assertTrue(chip is not None)

    @unittest.skipUnless(Path("/wang").exists(),
                         "Mock file is only available on Electronic "
                         "Vision(s) cluster")
    def test_chip_from_file(self):
        path = "/wang/data/calibration/hicann-dls-sr-hx/zeromock/" \
            "stable/latest/spiking_cocolist.pbin"
        chip = pynn.helper.chip_from_file(path)
        self.assertTrue(chip is not None)


if __name__ == '__main__':
    unittest.main()
