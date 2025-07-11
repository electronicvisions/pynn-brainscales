#!/usr/bin/env python

import unittest
import numpy as np
import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales import parameters


class TestSpikeSources(unittest.TestCase):

    def setUp(self):
        pynn.setup()

        self.pops_in = {}

        # SpikeSourceArray
        self.spike_times = [0.01, 0.03, 0.05, 0.07, 0.09]
        self.spike_times2 = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]

        self.pops_in['in1'] = pynn.Population(
            1,
            pynn.cells.SpikeSourceArray(spike_times=self.spike_times)
        )
        self.pops_in['npin1'] = pynn.Population(
            1,
            pynn.cells.SpikeSourceArray(spike_times=np.array(self.spike_times))
        )
        self.pops_in['sein1'] = pynn.Population(
            1,
            pynn.cells.SpikeSourceArray(
                spike_times=parameters.Sequence(self.spike_times))
        )

        self.pops_in['in2'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(spike_times=self.spike_times)
        )
        self.pops_in['npin2'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(spike_times=np.array(self.spike_times))
        )
        self.pops_in['sein2'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=parameters.Sequence(self.spike_times))
        )

        self.pops_in['in2var'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=[self.spike_times, self.spike_times2])
        )
        self.pops_in['npin2var'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=[np.array(self.spike_times),
                             np.array(self.spike_times2)])
        )
        self.pops_in['sein2var'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=[parameters.Sequence(self.spike_times),
                             parameters.Sequence(self.spike_times2)])
        )

        self.pops_in['in2equal'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(spike_times=[self.spike_times] * 2)
        )
        self.pops_in['npin2equal'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=[np.array(self.spike_times)] * 2)
        )
        self.pops_in['sein2equal'] = pynn.Population(
            2,
            pynn.cells.SpikeSourceArray(
                spike_times=[parameters.Sequence(self.spike_times)] * 2)
        )

        self.pops_in['mixedtypein3equal'] = pynn.Population(
            3,
            pynn.cells.SpikeSourceArray(
                spike_times=[
                    self.spike_times,
                    parameters.Sequence(self.spike_times),
                    np.array(self.spike_times)
                ])
        )
        self.pops_in['mixedtypein3var'] = pynn.Population(
            3,
            pynn.cells.SpikeSourceArray(
                spike_times=[
                    self.spike_times,
                    parameters.Sequence(self.spike_times2),
                    np.array(self.spike_times)])
        )

        # SpikeSourcePoisson
        poisson_properties = {"rate": 1e4, "start": 0, "duration": 10}
        self.pops_in['poisson1'] = pynn.Population(
            1,
            pynn.cells.SpikeSourcePoisson(**poisson_properties)
        )
        self.pops_in['poisson2'] = pynn.Population(
            2,
            pynn.cells.SpikeSourcePoisson(**poisson_properties)
        )

        # Target Populations
        self.pops = []
        self.pops.append(pynn.Population(1, pynn.cells.HXNeuron()))
        self.pops.append(pynn.Population(2, pynn.cells.HXNeuron()))

    def tearDown(self):
        pynn.run(None, pynn.RunCommand.PREPARE)
        pynn.end()

    # FIXME connector related tests need to be separated due to issue #3874
    def test_poppop1to1(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pop_in, pop, pynn.OneToOneConnector())

    def test_viewpop1to1(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pynn.PopulationView(pop_in, [0]),
                                pop,
                                pynn.OneToOneConnector())

    def test_popview1to1(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pop_in,
                                pynn.PopulationView(pop, [0]),
                                pynn.OneToOneConnector())

    def test_poppopalltoall(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pop_in, pop, pynn.AllToAllConnector())

    def test_viewpopalltoall(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pynn.PopulationView(pop_in, [0]),
                                pop,
                                pynn.AllToAllConnector())

    def test_popviewalltoall(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pop_in,
                                pynn.PopulationView(pop, [0]),
                                pynn.AllToAllConnector())

    @unittest.skip("This throws a c++ assert, see issue #3874")
    def test_identicalprojection(self):
        for pop in self.pops:
            for pop_in in self.pops_in.values():
                pynn.Projection(pop_in, pop, pynn.OneToOneConnector())
                pynn.Projection(pynn.PopulationView(pop_in, [0]),
                                pop,
                                pynn.OneToOneConnector())

    def test_accessors(self):
        pops_in = self.pops_in

        # SpikeSourceArray
        self.assertEqual(
            pops_in['in1'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['in2'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['npin2'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['sein2'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertTrue(np.array_equal(
            pops_in['in2var'].get("spike_times"),
            [parameters.Sequence(self.spike_times),
             parameters.Sequence(self.spike_times2)]
        ))
        self.assertTrue(np.array_equal(
            pops_in['npin2var'].get("spike_times"),
            [parameters.Sequence(self.spike_times),
             parameters.Sequence(self.spike_times2)]
        ))
        self.assertTrue(np.array_equal(
            pops_in['sein2var'].get("spike_times"),
            [parameters.Sequence(self.spike_times),
             parameters.Sequence(self.spike_times2)]
        ))
        self.assertEqual(
            pops_in['in2equal'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['npin2equal'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['sein2equal'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertEqual(
            pops_in['mixedtypein3equal'].get("spike_times"),
            parameters.Sequence(self.spike_times)
        )
        self.assertTrue(np.array_equal(
            pops_in['mixedtypein3var'].get("spike_times"),
            [parameters.Sequence(self.spike_times),
             parameters.Sequence(self.spike_times2),
             parameters.Sequence(self.spike_times)]
        ))

        my_spike_times = [1, 2, 3]
        pops_in['in1'].set(spike_times=my_spike_times)
        self.assertEqual(
            pops_in['in1'].get("spike_times"),
            parameters.Sequence(my_spike_times)
        )
        my_spike_times2 = [4, 5, 6]
        pops_in['in2'].set(spike_times=[my_spike_times, my_spike_times2])
        self.assertTrue(np.array_equal(
            pops_in['in2'].get("spike_times"),
            [parameters.Sequence(my_spike_times),
             parameters.Sequence(my_spike_times2)]
        ))


if __name__ == '__main__':
    unittest.main()
