#!/usr/bin/env python

import unittest
import numpy as np
import pygrenade_vx.network as grenade
from dlens_vx_v3 import halco
import pynn_brainscales.brainscales2 as pynn


class TestPlasticityRule(unittest.TestCase):

    def setUp(self):
        pynn.setup()

    def tearDown(self):
        pynn.end()

    def test_projection(self):
        pop1 = pynn.Population(3, pynn.cells.HXNeuron())
        pop2 = pynn.Population(4, pynn.cells.HXNeuron())

        timer = pynn.Timer(start=5, period=10, num_periods=1)
        synapse = pynn.standardmodels.synapses.PlasticSynapse(
            weight=63, plasticity_rule=pynn.plasticity_rules.PlasticityRule(
                timer=timer))

        proj = pynn.Projection(
            pop1, pop2, pynn.AllToAllConnector(), synapse_type=synapse)

        self.assertTrue(isinstance(
            proj.synapse_type,
            pynn.plasticity_rules.PlasticityRuleHandle))

        handle = proj.synapse_type.to_plasticity_rule_projection_handle(proj)

        expectation = grenade.ProjectionDescriptor()
        self.assertEqual(handle, expectation)

        pynn.run(None)

    def test_population_no_readout(self):
        timer = pynn.Timer(start=5, period=10, num_periods=1)
        neuron = pynn.cells.PlasticHXNeuron(
            plasticity_rule=pynn.plasticity_rules.PlasticityRule(timer=timer))

        pop = pynn.Population(3, neuron)

        self.assertTrue(isinstance(
            pop.celltype,
            pynn.plasticity_rules.PlasticityRuleHandle))

        handle = pop.celltype.to_plasticity_rule_population_handle(pop)

        expectation = \
            grenade.PlasticityRule.PopulationHandle()
        expectation.neuron_readout_sources = [
            {halco.CompartmentOnLogicalNeuron(): [None]},
            {halco.CompartmentOnLogicalNeuron(): [None]},
            {halco.CompartmentOnLogicalNeuron(): [None]}]
        self.assertEqual(handle, expectation)

        pynn.run(None)

    def test_population_readout(self):
        timer = pynn.Timer(start=5, period=10, num_periods=1)
        neuron = pynn.cells.PlasticHXNeuron(
            plasticity_rule_enable_readout_source=True,
            plasticity_rule_readout_source=pynn.cells.PlasticHXNeuron
            .ReadoutSource.membrane,
            plasticity_rule=pynn.plasticity_rules.PlasticityRule(timer=timer))

        pop = pynn.Population(3, neuron)

        self.assertTrue(isinstance(
            pop.celltype,
            pynn.plasticity_rules.PlasticityRuleHandle))

        handle = pop.celltype.to_plasticity_rule_population_handle(pop)

        expectation = \
            grenade.PlasticityRule.PopulationHandle()
        expectation.neuron_readout_sources = [
            {halco.CompartmentOnLogicalNeuron(): [
             pynn.cells.PlasticHXNeuron.ReadoutSource.membrane]}
        ] * len(pop)

        self.assertEqual(handle, expectation)

        self.assertTrue(
            np.array_equal(pop.get(
                "plasticity_rule_enable_readout_source", simplify=False),
                [True, True, True]))

        self.assertEqual(
            [pynn.cells.PlasticHXNeuron.ReadoutSource(int(e)) for e in pop.get(
                "plasticity_rule_readout_source", simplify=False)],
            [pynn.cells.PlasticHXNeuron.ReadoutSource.membrane] * len(pop))

        expectation = [
            pynn.cells.PlasticHXNeuron.ReadoutSource.membrane,
            pynn.cells.PlasticHXNeuron.ReadoutSource.adaptation,
            pynn.cells.PlasticHXNeuron.ReadoutSource.exc_synin,
        ]

        pop.set(plasticity_rule_readout_source=expectation)

        self.assertEqual(
            [pynn.cells.PlasticHXNeuron.ReadoutSource(int(rs)) for rs in
             pop.get("plasticity_rule_readout_source", simplify=False)],
            expectation)

        pynn.run(None)

    def test_population_projection(self):
        timer = pynn.Timer(start=5, period=10, num_periods=1)
        plasticity_rule = pynn.plasticity_rules.PlasticityRule(
            timer=timer)

        neuron = pynn.cells.PlasticHXNeuron(
            plasticity_rule_enable_readout_source=True,
            plasticity_rule_readout_source=pynn.cells.PlasticHXNeuron
            .ReadoutSource.membrane,
            plasticity_rule=plasticity_rule)

        synapse = pynn.standardmodels.synapses.PlasticSynapse(
            weight=63, plasticity_rule=plasticity_rule)

        pop1 = pynn.Population(3, pynn.cells.HXNeuron())
        pop = pynn.Population(4, neuron)

        proj = pynn.Projection(
            pop1, pop, pynn.AllToAllConnector(), synapse_type=synapse)

        self.assertTrue(isinstance(
            pop.celltype,
            pynn.plasticity_rules.PlasticityRuleHandle))

        self.assertTrue(isinstance(
            proj.synapse_type,
            pynn.plasticity_rules.PlasticityRuleHandle))

        pynn.run(None)


if __name__ == '__main__':
    unittest.main()
