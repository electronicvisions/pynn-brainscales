import unittest

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.cells import OffChipSource


# def _setup_spikeio_output(output_populations):
#     """ Setups the TX Pathway of the SpikeIO experiment, currently still
#      needed due to software limitations in grenade."""
#
#     pynn.run(None, pynn.RunCommand.PREPARE)
#     pynn.reset()
#
#     graph = simulator.state.grenade_network_graph.graph_translation. \
#         execution_instances[ExecutionInstanceID()]
#
#     def labels_for(pop):
#         gpop = PopulationOnExecutionInstance(simulator.state.
#                                              populations.index(pop))
#         labs = []
#         for comp_map in graph.spike_labels[gpop]:
#             for comp_labels in comp_map.values():
#                 labs += [lab for lab in comp_labels if lab is not None]
#         return labs
#
#     used = []
#     for population in graph.spike_labels.values():
#         for comp_map in population:
#             for comp_labels in comp_map.values():
#                 used += [lab for lab in comp_labels if lab is not None]
#
#     out_labels = []
#     for pop in output_populations:
#         out_labels += labels_for(pop)
#
#     builder = PlaybackProgramBuilder()
#
#     for lbl in used:
#         builder.write(
#             SpikeIOOutputRouteOnFPGA(lbl),
#             SpikeIOOutputRoute(SpikeIOOutputRoute.SILENT)
#         )
#
#     for idx, lbl in enumerate(out_labels):
#         builder.write(SpikeIOOutputRouteOnFPGA(lbl),
#                       SpikeIOOutputRoute(idx))
#
#     simulator.state.injection_pre_realtime = builder


class TestSpikeIOOffChipSourceBehaviour(unittest.TestCase):
    def setUp(self):
        pynn.setup()

    def tearDown(self):
        pynn.end()

    def _run_network(
            self,
            *,
            loopback_enabled: bool = True,
            rate_hz: float = 1e3,
            label: list[int] = None,
    ):

        tx_pop = pynn.Population(
            1, pynn.cells.SpikeSourcePoissonOnChip(rate=rate_hz, seed=53)
        )
        tx_pop.record("spikes")

        rx_virtual = pynn.Population(
            1,
            OffChipSource(
                enable_internal_loopback=loopback_enabled, data_rate_scaler=1,
                label=label
            ),
        )
        rx_virtual.record("spikes")

        # _setup_spikeio_output(
        #     output_populations=[tx_pop],
        # )

        pynn.run(10.0)

        spikes_tx = tx_pop.get_data("spikes").segments[-1].spiketrains
        spikes_rx = rx_virtual.get_data("spikes").segments[-1].spiketrains

        tx_count = sum(len(st) for st in spikes_tx)
        rx_count = sum(len(st) for st in spikes_rx)

        return tx_count, rx_count, spikes_tx, spikes_rx

    @unittest.skip("Disabled until feature integrated again")
    def test_source_requires_labels(self):
        with self.assertRaises(ValueError):
            OffChipSource(
                enable_internal_loopback=True, data_rate_scaler=1, label=None)
        with self.assertRaises(ValueError):
            OffChipSource(
                enable_internal_loopback=True, data_rate_scaler=1, label=[])

    @unittest.skip("Disabled until feature integrated again")
    def test_offchip_address_no_spikes(self):
        tx_count, rx_count, _, _ = self._run_network(
            loopback_enabled=True,
            rate_hz=1e3,
            label=[10],
        )
        self.assertGreater(tx_count, 0)
        self.assertEqual(rx_count, 0)

    @unittest.skip("Disabled until feature integrated again")
    def test_loopback_disabled(self):
        """ Loopback = False should result in no spikes received back."""
        tx_count, rx_count, _, _ = self._run_network(
            loopback_enabled=False,
            rate_hz=1e3,
            label=[0],
        )
        self.assertGreater(tx_count, 0)
        self.assertEqual(rx_count, 0)

    @unittest.skip("Disabled until feature integrated again")
    def test_loopback_enabled(self):
        tx_count, rx_count, _, _ = self._run_network(
            loopback_enabled=True,
            rate_hz=1e3,
            label=[0],
        )
        self.assertGreater(tx_count, 0)
        self.assertGreater(rx_count, 0)


if __name__ == "__main__":
    unittest.main()
