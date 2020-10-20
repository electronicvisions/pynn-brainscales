import unittest


class TestSingleNeuronDemo(unittest.TestCase):
    @staticmethod
    def test_script():
        # pylint: disable=import-outside-toplevel,unused-import
        import examples.single_neuron_demo


if __name__ == "__main__":
    unittest.main()
