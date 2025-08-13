import unittest
import numpy as np
from src.connectivity.graph_builder import GraphBuilder
from src.connectivity.matrix_processor import MatrixProcessor

class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.matrix = np.random.rand(21, 21)
        self.matrix = (self.matrix + self.matrix.T) / 2  # Make it symmetric
        self.processor = MatrixProcessor(self.matrix)
        self.processor.process_matrix()
        self.graph_builder = GraphBuilder(self.processor)

    def test_graph_edges_count(self):
        self.graph_builder.build_graph()
        edges = self.graph_builder.get_edges()
        expected_edge_count = int(np.count_nonzero(self.processor.get_upper_triangle()) * 0.5)
        self.assertEqual(len(edges), expected_edge_count)

    def test_graph_node_identifiers(self):
        self.graph_builder.build_graph()
        nodes = self.graph_builder.get_nodes()
        self.assertEqual(len(nodes), 21)  # There should be 21 nodes

    def test_graph_edge_weights(self):
        self.graph_builder.build_graph()
        edges = self.graph_builder.get_edges()
        for edge in edges:
            self.assertGreater(edge[2], 0)  # Ensure all retained edges have positive weights

if __name__ == '__main__':
    unittest.main()