import unittest
import numpy as np
from src.connectivity.matrix_processor import MatrixProcessor

class TestMatrixProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = MatrixProcessor()
        self.test_matrix = np.random.rand(21, 21)
        self.test_matrix = (self.test_matrix + self.test_matrix.T) / 2  # Make it symmetric

    def test_extract_upper_triangle(self):
        upper_triangle = self.processor.extract_upper_triangle(self.test_matrix)
        self.assertEqual(upper_triangle.shape, (21, 21))
        self.assertTrue(np.all(upper_triangle == np.triu(self.test_matrix)))

    def test_get_top_edges(self):
        upper_triangle = self.processor.extract_upper_triangle(self.test_matrix)
        top_edges = self.processor.get_top_edges(upper_triangle)
        expected_num_edges = int(np.count_nonzero(np.triu(upper_triangle)) / 2)
        self.assertEqual(len(top_edges), expected_num_edges // 2)

    def test_process_matrix(self):
        top_edges = self.processor.process_matrix(self.test_matrix)
        self.assertIsInstance(top_edges, list)
        self.assertGreater(len(top_edges), 0)

if __name__ == '__main__':
    unittest.main()