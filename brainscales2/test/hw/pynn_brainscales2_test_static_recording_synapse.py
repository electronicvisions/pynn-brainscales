#!/usr/bin/env python

import unittest
from pynn_brainscales.brainscales2.examples.static_recording_synapse import \
    main


class TestStaticRecordingSynapse(unittest.TestCase):
    def test_main(self):
        weights = main()
        self.assertEqual(len(weights), 1)
        self.assertGreaterEqual(weights[0].time, 4 * 125 * 1000)
        self.assertLessEqual(weights[0].time, 6 * 125 * 1000)
        self.assertEqual(len(weights[0].data), 1)
        self.assertEqual(weights[0].data[0], [63])


if __name__ == "__main__":
    unittest.main()
