#!/usr/bin/env python

from typing import List, Tuple
import unittest

import numpy as np
import neo

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2.morphology import create_mc_neuron, \
    Compartment, SharedLineConnection, McNeuronBase
from pynn_brainscales.brainscales2.examples.multicompartment import main


class TestMulticompartmentExample(unittest.TestCase):
    @staticmethod
    def test_main():
        # Simply tests if program runs
        main()


class TestRecordingAndProjections(unittest.TestCase):
    '''
    Create a chain of compartments and inject an external input in one
    compartment after another. Assert that the EPSP at the injection site is
    the largest.
    '''

    @staticmethod
    def get_psp_heights(signal: neo.IrregularlySampledSignal, n_inputs: int
                        ) -> List:
        '''
        Extract the PSP heights from a signal with several inputs.

        :param signal: Recording from which to extract the PSP heights.
        :param n_inputs: Number of inputs. The signal is cut in n_inputs equal
            segments and in each segment the PSP height is calculated.
        :return: PSP height for the individual inputs. The PSP height is
            calculated as the difference between the maximum and the mean
            voltage of the first twenty data points.
        '''
        voltage = signal.magnitude[:, 0]
        points_per_input = len(voltage) // n_inputs

        psp_heights = []
        for i in range(n_inputs):
            cut_v = voltage[i * points_per_input:(i + 1) * points_per_input]

            psp_heights.append(np.max(cut_v) - np.mean(cut_v[:20]))

        return psp_heights

    @staticmethod
    def create_chain() -> Tuple[McNeuronBase, List[str]]:
        """
        Create a neuron class which represents a chain with
        three compartments.

        :return: Neuron class and labels for different compartments.
        """
        labels = [f'comp_{i}' for i in range(3)]
        comps = []
        comps.append(Compartment(positions=[0], label=labels[0],
                                 connect_shared_line=[0]))
        comps.append(Compartment(positions=[1, 2], label=labels[1],
                                 connect_conductance=[(1, 1000)],
                                 connect_shared_line=[2]))
        comps.append(Compartment(positions=[3, 4], label=labels[2],
                                 connect_conductance=[(3, 1000)]))

        connections = [SharedLineConnection(start=0, stop=1, row=0),
                       SharedLineConnection(start=2, stop=3, row=0)]

        McNeuron = create_mc_neuron(
            'McNeuron', compartments=comps,
            connections=connections, single_active_circuit=True)
        return McNeuron, labels

    def test_recording_and_projections(self):
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

        McNeuron, labels = self.create_chain()
        pop = pynn.Population(1, McNeuron(threshold_enable=False))

        isi = 0.5   # ms (hw)
        spikes = (np.arange(len(labels)) + 0.5) * isi

        for label, spike in zip(labels, spikes):
            source = pynn.Population(5, pynn.cells.SpikeSourceArray(
                spike_times=[spike]))
            pynn.Projection(source, pop,
                            pynn.AllToAllConnector(location_selector=label),
                            synapse_type=StaticSynapse(weight=63))

        # record each compartment once
        psp_heights = []
        for label in labels:
            pop.record(None)
            pop.record(['v'], locations=[label])

            pynn.run(isi * len(labels))

            data = pop.get_data(clear=True).segments[-1]\
                .irregularlysampledsignals[-1]
            psp_heights.append(self.get_psp_heights(data, len(labels)))
            pynn.reset()

        self.assertTrue(
            np.all(np.argmax(psp_heights, axis=1) == np.arange(len(labels))))
        pynn.end()

    def test_cadc_recording(self):
        """
        Test that samples are recorded for all requested compartments.
        """
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

        McNeuron, labels = self.create_chain()
        pop = pynn.Population(1, McNeuron(threshold_enable=False))

        runtime = 0.15
        spikes = np.linspace(0.01, runtime - 0.05, len(labels))

        for label, spike in zip(labels, spikes):
            source = pynn.Population(5, pynn.cells.SpikeSourceArray(
                spike_times=[spike]))
            pynn.Projection(source, pop,
                            pynn.AllToAllConnector(location_selector=label),
                            synapse_type=StaticSynapse(weight=63))

        pop.record("v", locations=labels, device='cadc')
        pynn.run(runtime)

        traces = pop.get_data("v").segments[0].irregularlysampledsignals

        # One recorded trace for each compartment
        self.assertEqual(len(traces), len(labels))
        for trace in traces:
            self.assertTrue(trace.size > 0)

        # Check that traces differ
        self.assertFalse(np.all(traces[0] == traces[1]))

        pynn.end()

    def test_readout_source(self):
        """
        Test that the selection of the readout source works.
        """
        pynn.setup(initial_config=pynn.helper.chip_from_nightly())

        McNeuron, labels = self.create_chain()
        pop = pynn.Population(1, McNeuron(threshold_enable=False))

        in_pop = pynn.Population(5, pynn.cells.SpikeSourceArray(
            spike_times=[0.5, 0.6]))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=63)
        pynn.Projection(in_pop, pop, pynn.AllToAllConnector(),
                        synapse_type=synapse)
        pop.record("v", locations=[labels[0]])
        pynn.run(1)
        pynn.reset()

        samples = pop.get_data().segments[-1].irregularlysampledsignals[0]

        # check that input spikes have an effect.
        self.assertGreater(samples.max() - samples[:100].mean(), 20)

        # test that changing the readout source works (synaptic inputs
        # pull the synaptic line down)
        pop.record(None)
        pop.record("exc_synin", locations=[labels[0]])
        pynn.run(1)
        samples = pop.get_data().segments[-1].irregularlysampledsignals[0]

        # also assert that the maximum is not too high. Otherwise we
        # could still record the membrane and detect the reset.
        self.assertLess(samples.max() - samples[:100].mean(), 10)
        self.assertGreater(samples[:100].mean() - samples.min(), 20)

        pynn.end()


if __name__ == "__main__":
    unittest.main()
