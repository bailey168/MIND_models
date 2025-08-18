import numpy as np
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
from typing import Tuple, Optional

def connectivity_matrix_to_graph(
    connectivity_matrix: np.ndarray,
    top_percentile: float = 0.5,
    use_absolute_values: bool = True
) -> Data:
    """
    Convert a symmetric brain connectivity matrix to a PyTorch Geometric graph.
    
    Args:
        connectivity_matrix: Symmetric matrix where each row/column represents a brain region
        top_percentile: Fraction of edges to keep (e.g., 0.5 for top 50%)
        use_absolute_values: Whether to use absolute values for edge selection
        
    Returns:
        PyTorch Geometric Data object representing the brain connectivity graph
    """
    assert connectivity_matrix.shape[0] == connectivity_matrix.shape[1], "Matrix must be square"
    assert 0 < top_percentile <= 1, "top_percentile must be between 0 and 1"
    
    n_regions = connectivity_matrix.shape[0]
    
    # Extract upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n_regions, k=1)
    upper_tri_values = connectivity_matrix[upper_tri_indices]
    
    # Determine threshold for top edges
    values_for_threshold = np.abs(upper_tri_values) if use_absolute_values else upper_tri_values
    threshold = np.percentile(values_for_threshold, (1 - top_percentile) * 100)
    
    # Select edges above threshold
    if use_absolute_values:
        selected_mask = np.abs(upper_tri_values) >= threshold
    else:
        selected_mask = upper_tri_values >= threshold
    
    # Get selected edge indices and values
    selected_edges_i = upper_tri_indices[0][selected_mask]
    selected_edges_j = upper_tri_indices[1][selected_mask]
    selected_edge_weights = upper_tri_values[selected_mask]
    
    # Create bidirectional edges (since original matrix was symmetric)
    edge_index = torch.tensor([
        np.concatenate([selected_edges_i, selected_edges_j]),
        np.concatenate([selected_edges_j, selected_edges_i])
    ], dtype=torch.long)
    
    # Create bidirectional edge weights
    edge_attr = torch.tensor(
        np.concatenate([selected_edge_weights, selected_edge_weights]), 
        dtype=torch.float32
    ).unsqueeze(1)
    
    # Create one-hot encoded node features (brain region identity)
    node_features = torch.eye(n_regions, dtype=torch.float32)
    
    # Create the graph data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_regions
    )
    
    return data

def load_connectivity_matrix(file_path: str) -> np.ndarray:
    """
    Load connectivity matrix from various file formats.
    
    Args:
        file_path: Path to the connectivity matrix file
        
    Returns:
        Loaded connectivity matrix as numpy array
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        # Assume the matrix is stored under 'connectivity' key or first array
        if 'connectivity' in data:
            return data['connectivity']
        else:
            return data[data.files[0]]
    elif file_path.endswith('.txt') or file_path.endswith('.csv'):
        delimiter = ',' if file_path.endswith('.csv') else None
        return np.loadtxt(file_path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_graph_data(data: Data, output_path: str):
    """
    Save the graph data object.
    
    Args:
        data: PyTorch Geometric Data object
        output_path: Path where to save the graph data
    """
    if output_path.endswith('.pt'):
        torch.save(data, output_path)
    elif output_path.endswith('.pkl'):
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        # Default to .pt format
        torch.save(data, output_path + '.pt')

def analyze_graph_properties(data: Data) -> dict:
    """
    Analyze basic properties of the created graph.
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        Dictionary containing graph statistics
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2  # Divide by 2 since edges are bidirectional
    
    # Calculate node degrees
    edge_index = data.edge_index
    degrees = torch.zeros(num_nodes)
    for i in range(num_nodes):
        degrees[i] = (edge_index[0] == i).sum().item()
    
    properties = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': degrees.mean().item(),
        'max_degree': degrees.max().item(),
        'min_degree': degrees.min().item(),
        'density': (2 * num_edges) / (num_nodes * (num_nodes - 1)),
        'edge_weight_stats': {
            'mean': data.edge_attr.mean().item(),
            'std': data.edge_attr.std().item(),
            'min': data.edge_attr.min().item(),
            'max': data.edge_attr.max().item()
        }
    }
    
    return properties

# Example usage and main function
def main():
    """
    Example usage of the brain connectivity to graph conversion.
    """
    # Example: Create a synthetic connectivity matrix for demonstration
    n_regions = 100  # Number of brain regions
    np.random.seed(42)
    
    # Create a symmetric connectivity matrix
    random_matrix = np.random.randn(n_regions, n_regions)
    connectivity_matrix = (random_matrix + random_matrix.T) / 2
    np.fill_diagonal(connectivity_matrix, 0)  # No self-connections
    
    print(f"Original connectivity matrix shape: {connectivity_matrix.shape}")
    print(f"Number of possible edges: {n_regions * (n_regions - 1) // 2}")
    
    # Convert to graph keeping top 50% of edges
    graph_data = connectivity_matrix_to_graph(
        connectivity_matrix, 
        top_percentile=0.5,
        use_absolute_values=True
    )
    
    # Analyze graph properties
    properties = analyze_graph_properties(graph_data)
    
    print("\nGraph Properties:")
    for key, value in properties.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value:.4f}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Save the graph data
    save_graph_data(graph_data, "brain_connectivity_graph.pt")
    print(f"\nGraph saved to: brain_connectivity_graph.pt")
    
    return graph_data

if __name__ == "__main__":
    # Run the example
    graph_data = main()
    
    # If you have your own connectivity matrix file, use this instead:
    # connectivity_matrix = load_connectivity_matrix("path/to/your/matrix.npy")
    # graph_data = connectivity_matrix_to_graph(connectivity_matrix, top_percentile=0.5)
    # save_graph_data(graph_data, "your_brain_graph.pt")