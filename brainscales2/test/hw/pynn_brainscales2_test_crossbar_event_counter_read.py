#!/usr/bin/env python

import unittest
from pynn_brainscales.brainscales2.examples.crossbar_event_counter_read \
    import main
from dlens_vx_v2 import halco


class TestCrossbarEventCounterRead(unittest.TestCase):
    def test_main(self):
        """
        Assert, that the expected number of input and output spikes is
        recorded.
        """
        counts = main(num_spikes=200, runtime=20.)
        # input spikes
        self.assertLess(counts[halco.CrossbarOutputOnDLS(0)], 240)
        self.assertGreater(counts[halco.CrossbarOutputOnDLS(0)], 160)
        # output spikes
        self.assertLess(counts[halco.CrossbarOutputOnDLS(8)], 240)
        self.assertGreater(counts[halco.CrossbarOutputOnDLS(8)], 160)


if __name__ == "__main__":
    unittest.main()
