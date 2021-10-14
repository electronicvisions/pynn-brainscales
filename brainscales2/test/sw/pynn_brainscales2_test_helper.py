#!/usr/bin/env python

import os
import tempfile
import unittest
from unittest import mock
import numpy
import pynn_brainscales.brainscales2 as pynn
from dlens_vx_v2 import halco, hal, lola, sta


class TestHelper(unittest.TestCase):

    def test_coco_extraction(self):
        builder = sta.PlaybackProgramBuilderDumper()
        an_coord0 = halco.AtomicNeuronOnDLS(halco.common.Enum(0))
        an_coord1 = halco.AtomicNeuronOnDLS(halco.common.Enum(1))
        neuron0 = lola.AtomicNeuron()
        neuron0.leak.i_bias = 666
        neuron1 = lola.AtomicNeuron()
        neuron1.leak.i_bias = 420
        builder.write(an_coord0, neuron0)
        builder.write(an_coord1, neuron1)

        common_config = hal.CommonNeuronBackendConfig()
        common_config.clock_scale_fast = 3
        common_coord = halco.CommonNeuronBackendConfigOnDLS()
        builder.write(common_coord, common_config)

        full_coco = {}
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "dump")
            with open(filename, "wb") as fd:
                fd.write(sta.to_portablebinary(builder.done()))
            full_coco = pynn.helper.coco_from_file(filename)
        self.assertTrue(an_coord0 in full_coco)
        self.assertTrue(an_coord1 in full_coco)
        hx_coco = pynn.helper.filter_atomic_neuron(full_coco)
        self.assertTrue(an_coord0 in hx_coco)
        self.assertTrue(an_coord1 in hx_coco)
        self.assertFalse(common_coord in hx_coco)
        remainder_coco = pynn.helper.filter_non_atomic_neuron(full_coco)
        self.assertFalse(an_coord0 in remainder_coco)
        self.assertTrue(common_coord in remainder_coco)
        self.assertEqual(remainder_coco[common_coord].clock_scale_fast, 3)

        pynn.setup()
        pop = pynn.Population(2, pynn.cells.HXNeuron(hx_coco))
        self.assertTrue(
            numpy.array_equal(pop.get("leak_i_bias"), [666, 420]))
        pynn.run(None)
        pynn.end()

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_nightly_calib_path(self):
        expected_path = "/wang/data/calibration/hicann-dls-sr-hx/zeromock/" \
            "stable/latest/spiking_cocolist.pbin"
        tested_path = pynn.helper.nightly_calib_path()
        self.assertEqual(expected_path, str(tested_path))

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_nightly_coco_extraction(self):
        atomic, inject = pynn.helper.filtered_cocos_from_nightly()
        self.assertTrue(atomic is not None)
        self.assertTrue(inject is not None)


if __name__ == '__main__':
    unittest.main()
