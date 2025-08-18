import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def connectivity_matrix_to_graph(connectivity_matrix, keep_top_percent=0.5, add_self_loops_flag=True, region_names=None):
    """
    Convert a symmetric connectivity matrix to a graph.
    
    Args:
        connectivity_matrix: NxN symmetric numpy array or torch tensor
        keep_top_percent: fraction of edges to keep (default 0.5 for top 50%)
        add_self_loops_flag: whether to add self-loops to nodes
        region_names: list of strings with region names (length N), or None for default names
        
    Returns:
        PyTorch Geometric Data object with:
        - x: one-hot encoded node identities (NxN)
        - edge_index: sparse edge representation
        - edge_attr: connectivity strength values
        - pos: node positions in circular layout
        - region_names: list of region names
        - region_to_idx: dictionary mapping region names to node indices
        - idx_to_region: dictionary mapping node indices to region names
    """
    
    # Convert to numpy if torch tensor
    if torch.is_tensor(connectivity_matrix):
        conn_matrix = connectivity_matrix.numpy()
    else:
        conn_matrix = connectivity_matrix.copy()
    
    # Ensure matrix is square and symmetric
    assert conn_matrix.ndim == 2, "Input must be a 2D matrix"
    assert conn_matrix.shape[0] == conn_matrix.shape[1], "Matrix must be square"
    assert np.allclose(conn_matrix, conn_matrix.T), "Matrix must be symmetric"
    
    num_nodes = conn_matrix.shape[0]
    
    # Handle region names
    if region_names is None:
        region_names = [f"Region_{i:03d}" for i in range(num_nodes)]
    else:
        assert len(region_names) == num_nodes, f"Number of region names ({len(region_names)}) must match matrix size ({num_nodes})"
        # Ensure all names are strings
        region_names = [str(name) for name in region_names]
    
    # Create mapping dictionaries
    region_to_idx = {name: idx for idx, name in enumerate(region_names)}
    idx_to_region = {idx: name for idx, name in enumerate(region_names)}
    
    # Extract upper triangle (excluding diagonal)
    upper_tri_mask = np.triu(np.ones_like(conn_matrix, dtype=bool), k=1)
    upper_tri_values = conn_matrix[upper_tri_mask]
    
    # Get indices of upper triangle
    upper_tri_indices = np.where(upper_tri_mask)
    
    # Calculate threshold for top percentage of edges
    num_edges_to_keep = int(len(upper_tri_values) * keep_top_percent)
    
    # Handle edge case where no edges would be kept
    if num_edges_to_keep == 0 and len(upper_tri_values) > 0:
        num_edges_to_keep = 1
    
    # Find top edges by absolute magnitude
    edge_magnitudes = np.abs(upper_tri_values)
    top_edge_indices = np.argsort(edge_magnitudes)[-num_edges_to_keep:]
    
    # Create edge list (bidirectional since graph is undirected)
    edge_list = []
    edge_weights = []
    edge_region_pairs = []  # Store region name pairs for each edge
    
    for idx in top_edge_indices:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        weight = conn_matrix[i, j]
        
        # Add both directions for undirected graph
        edge_list.extend([[i, j], [j, i]])
        edge_weights.extend([weight, weight])
        
        # Store region pairs (same pair for both directions)
        region_pair = (region_names[i], region_names[j])
        edge_region_pairs.extend([region_pair, region_pair])
    
    # Convert to torch tensors
    if edge_list:  # Check if we have edges
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    else:
        # Create empty edge tensors if no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
        edge_region_pairs = []
    
    # Create one-hot encoded node features (node identities)
    node_features = torch.eye(num_nodes, dtype=torch.float)  # NxN identity matrix
    
    # Add self-loops if requested
    if add_self_loops_flag:
        edge_index, edge_attr = add_self_loops(
            edge_index, 
            edge_attr, 
            fill_value=1.0,  # Self-loop weight
            num_nodes=num_nodes
        )
        
        # Add self-loop region pairs
        if edge_region_pairs or add_self_loops_flag:
            self_loop_pairs = [(region_names[i], region_names[i]) for i in range(num_nodes)]
            edge_region_pairs.extend(self_loop_pairs)
    
    # Create node positions using circular layout
    if num_nodes == 1:
        pos = torch.tensor([[0.0, 0.0]], dtype=torch.float)
    else:
        angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        pos = torch.tensor([[np.cos(angle), np.sin(angle)] for angle in angles], dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=num_nodes
    )
    
    # Add custom attributes for region information
    data.region_names = region_names
    data.region_to_idx = region_to_idx
    data.idx_to_region = idx_to_region
    data.edge_region_pairs = edge_region_pairs
    
    return data

def get_region_features_by_name(data, region_name):
    """
    Get node features for a specific region by name.
    
    Args:
        data: PyTorch Geometric Data object from connectivity_matrix_to_graph
        region_name: string name of the region
        
    Returns:
        torch.Tensor: node features for the specified region
    """
    if region_name not in data.region_to_idx:
        raise ValueError(f"Region '{region_name}' not found. Available regions: {list(data.region_to_idx.keys())}")
    
    idx = data.region_to_idx[region_name]
    return data.x[idx]

def get_edges_for_region(data, region_name):
    """
    Get all edges connected to a specific region.
    
    Args:
        data: PyTorch Geometric Data object from connectivity_matrix_to_graph
        region_name: string name of the region
        
    Returns:
        dict: containing edge indices, weights, and connected region names
    """
    if region_name not in data.region_to_idx:
        raise ValueError(f"Region '{region_name}' not found. Available regions: {list(data.region_to_idx.keys())}")
    
    region_idx = data.region_to_idx[region_name]
    
    # Find edges where this region is either source or target
    edge_mask = (data.edge_index[0] == region_idx) | (data.edge_index[1] == region_idx)
    connected_edges = data.edge_index[:, edge_mask]
    edge_weights = data.edge_attr[edge_mask]
    
    # Get connected region names
    connected_regions = []
    for i in range(connected_edges.shape[1]):
        src_idx, tgt_idx = connected_edges[0, i].item(), connected_edges[1, i].item()
        if src_idx == region_idx:
            connected_regions.append(data.idx_to_region[tgt_idx])
        else:
            connected_regions.append(data.idx_to_region[src_idx])
    
    return {
        'edge_indices': connected_edges,
        'edge_weights': edge_weights,
        'connected_regions': connected_regions
    }

def example_usage():
    """Example of how to use the connectivity matrix conversion function with region names"""
    
    # Example with brain region names
    brain_regions = [
        "Frontal_Cortex", "Parietal_Cortex", "Temporal_Cortex", "Occipital_Cortex",
        "Hippocampus", "Amygdala", "Thalamus", "Hypothalamus", "Brainstem",
        "Cerebellum", "Caudate", "Putamen", "Globus_Pallidus", "Nucleus_Accumbens",
        "Insula", "Cingulate_Cortex", "Precuneus", "Angular_Gyrus", "Fusiform_Gyrus",
        "Superior_Temporal_Gyrus", "Middle_Temporal_Gyrus"
    ]
    
    n_regions = len(brain_regions)
    print(f"Testing with {n_regions} brain regions")
    print("="*60)
    
    # Create a sample connectivity matrix
    np.random.seed(42)
    random_matrix = np.random.randn(n_regions, n_regions)
    connectivity_matrix = (random_matrix + random_matrix.T) / 2
    np.fill_diagonal(connectivity_matrix, 0)
    
    # Convert to graph with region names
    graph_data = connectivity_matrix_to_graph(
        connectivity_matrix, 
        keep_top_percent=0.3, 
        add_self_loops_flag=True,
        region_names=brain_regions
    )
    
    print(f"Graph properties:")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.edge_index.shape[1]}")
    print(f"Number of region names: {len(graph_data.region_names)}")
    
    print(f"\nFirst 5 region names:")
    for i, name in enumerate(graph_data.region_names[:5]):
        print(f"  {i}: {name}")
    
    # Test region-specific queries
    test_region = "Hippocampus"
    print(f"\nTesting region-specific functions for '{test_region}':")
    
    # Get features for specific region
    features = get_region_features_by_name(graph_data, test_region)
    print(f"Features shape for {test_region}: {features.shape}")
    
    # Get edges for specific region
    edge_info = get_edges_for_region(graph_data, test_region)
    print(f"Number of edges connected to {test_region}: {len(edge_info['connected_regions'])}")
    print(f"Connected regions: {edge_info['connected_regions'][:5]}...")  # Show first 5
    
    # Test with default region names
    print(f"\n" + "="*60)
    print("Testing with default region names")
    
    small_matrix = np.random.randn(5, 5)
    small_matrix = (small_matrix + small_matrix.T) / 2
    np.fill_diagonal(small_matrix, 0)
    
    graph_default = connectivity_matrix_to_graph(small_matrix, keep_top_percent=0.5)
    print(f"Default region names: {graph_default.region_names}")

def test_edge_cases_with_regions():
    """Test edge cases with region names"""
    
    print(f"\n{'='*50}")
    print("Testing edge cases with region names")
    print(f"{'='*50}")
    
    # Test with custom region names
    custom_regions = ["RegionA", "RegionB", "RegionC"]
    matrix_3x3 = np.array([
        [0, 0.8, 0.3],
        [0.8, 0, 0.6],
        [0.3, 0.6, 0]
    ])
    
    graph_custom = connectivity_matrix_to_graph(
        matrix_3x3, 
        keep_top_percent=0.67,  # Keep 2 out of 3 possible edges
        region_names=custom_regions
    )
    
    print(f"Custom regions: {graph_custom.region_names}")
    print(f"Region to index mapping: {graph_custom.region_to_idx}")
    print(f"Edge region pairs: {graph_custom.edge_region_pairs[:4]}...")  # First few edges

if __name__ == "__main__":
    # Run examples
    example_usage()
    test_edge_cases_with_regions()