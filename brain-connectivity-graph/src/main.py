from connectivity.matrix_processor import MatrixProcessor
from connectivity.graph_builder import GraphBuilder
from encoding.node_encoder import NodeEncoder

def main():
    # Example: Load a 21x21 symmetric matrix (this should be replaced with actual data loading)
    matrix = [[0.0] * 21 for _ in range(21)]  # Placeholder for the actual matrix

    # Process the matrix
    processor = MatrixProcessor(matrix)
    upper_triangle = processor.extract_upper_triangle()
    top_edges = processor.get_top_edges(percentage=50)

    # Build the graph
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(top_edges)

    # One-hot encode the node identities
    encoder = NodeEncoder()
    encoded_nodes = encoder.one_hot_encode(graph.nodes)

    # Output the results (this can be modified as needed)
    print("Graph Nodes (One-Hot Encoded):", encoded_nodes)
    print("Graph Edges:", graph.edges)

if __name__ == "__main__":
    main()