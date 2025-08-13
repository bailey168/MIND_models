import unittest
import numpy as np
from src.encoding.node_encoder import NodeEncoder

class TestNodeEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = NodeEncoder()
        self.brain_regions = ['Region1', 'Region2', 'Region3', 'Region4', 'Region5']
        self.one_hot_encoded = self.encoder.one_hot_encode(self.brain_regions)

    def test_one_hot_encoding(self):
        expected_output = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(self.one_hot_encoded, expected_output)

    def test_unique_encoding(self):
        unique_regions = list(set(self.brain_regions))
        encoded = self.encoder.one_hot_encode(unique_regions)
        self.assertEqual(encoded.shape[0], len(unique_regions))

if __name__ == '__main__':
    unittest.main()