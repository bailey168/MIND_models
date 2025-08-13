from typing import List, Tuple
import numpy as np
import networkx as nx

class GraphBuilder:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.graph = nx.Graph()

    def build_graph(self) -> None:
        upper_triangle_indices = np.triu_indices(self.matrix.shape[0], k=1)
        edge_weights = self.matrix[upper_triangle_indices]
        
        # Get the top 50% of edges by magnitude
        threshold = np.percentile(edge_weights, 50)
        top_edges = edge_weights[edge_weights >= threshold]
        
        for idx, weight in zip(zip(*upper_triangle_indices), edge_weights):
            if weight in top_edges:
                self.graph.add_edge(idx[0], idx[1], weight=weight)

    def get_graph(self) -> nx.Graph:
        return self.graph

    def get_edges(self) -> List[Tuple[int, int, float]]:
        return [(u, v, d['weight']) for u, v, d in self.graph.edges(data=True)]