import unittest


class TestSingleNeuronDemo(unittest.TestCase):
    @staticmethod
    def test_script():
        # pylint: disable=import-outside-toplevel,unused-import
        import pynn_brainscales.brainscales2.examples.single_neuron_demo


if __name__ == "__main__":
    unittest.main()
