import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

def load_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    # Load the full CSV including headers
    data = np.genfromtxt(path, delimiter=",", dtype=str)
    
    # Extract row labels (first column, skip first cell which might be empty or a corner label)
    row_labels = data[1:, 0].tolist()  # Skip header row, take first column
    
    # Extract the numeric matrix (skip first row and first column)
    mat = data[1:, 1:].astype(float)
    
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be a square 2D matrix")
    
    # Check if matrix is symmetric
    if not np.allclose(mat, mat.T, rtol=1e-10, atol=1e-10):
        raise ValueError("Input matrix must be symmetric")

    return mat, row_labels

def matrix_to_graph(mat: np.ndarray, keep_percent: float = 0.5, keep_self_loops: bool = False, node_names: list[str] = None) -> Data:
    # Basic checks
    if not (0 < keep_percent <= 1):
        raise ValueError("keep_percent must be in (0, 1]")
    n = mat.shape[0]

    # Make a copy of the matrix
    mat = np.array(mat, dtype=float)
    # Remove self-loops for edge selection
    np.fill_diagonal(mat, 0.0)

    # Work on upper triangle indices (i < j)
    iu, ju = np.triu_indices(n, k=1)
    vals = mat[iu, ju]

    # Handle NaNs by treating them as zero magnitude (won't be selected if keep_percent < 1)
    vals = np.nan_to_num(vals, nan=0.0)

    # Determine number of edges to keep
    total_upper = vals.size
    k = int(np.floor(keep_percent * total_upper))
    k = max(1, min(k, total_upper))

    # Get indices of top-k by absolute magnitude
    order = np.argsort(np.abs(vals))[::-1]  # descending by |val|
    top_idx = order[:k]

    # Selected undirected pairs
    src_u = iu[top_idx]
    dst_u = ju[top_idx]
    w_u = vals[top_idx]

    # Build edge_index with both directions to represent an undirected graph in PyG
    src = np.concatenate([src_u, dst_u], axis=0)
    dst = np.concatenate([dst_u, src_u], axis=0)
    w = np.concatenate([w_u, w_u], axis=0)

    # Use torch.from_numpy for zero-copy conversion
    edge_index_np = np.array([src, dst])
    edge_index = torch.from_numpy(edge_index_np).long()
    edge_attr = torch.from_numpy(w).float()

    # One-hot node features (brain region identity)
    x = torch.eye(n, dtype=torch.float32)

    # Optionally add diagonal entries (self-loops) as edges if requested
    if keep_self_loops:
        self_nodes = torch.arange(n, dtype=torch.long)
        self_edges = torch.stack([self_nodes, self_nodes], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        # Self-loop weights set to 0.0 to avoid overpowering
        edge_attr = torch.cat([edge_attr, torch.zeros(n, dtype=torch.float32)], dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
    
    # Add node names if provided
    if node_names is not None:
        if len(node_names) != n:
            raise ValueError(f"Number of node names ({len(node_names)}) must match matrix size ({n})")
        data.node_names = node_names

    return data

def save_edgelist_csv(path: Path, edge_index: torch.Tensor, edge_attr: torch.Tensor):
    # For quick inspection/debugging
    rows = torch.stack([edge_index[0], edge_index[1], edge_attr], dim=1).cpu().numpy()
    header = "source,target,weight"
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt=["%d", "%d", "%.6f"])

def main():
    parser = argparse.ArgumentParser(description="Convert a symmetric matrix to a PyTorch Geometric graph.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to matrix (.csv)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .pt file to save the graph Data object")
    parser.add_argument("--keep-percent", "-p", type=float, default=0.5, help="Percentage of top-magnitude edges to keep from the upper triangle (0-1]")
    parser.add_argument("--self-loops", action="store_true", help="Include self-loops with weight 0.0")
    parser.add_argument("--export-edgelist", type=str, default="", help="Optional CSV path to export (source,target,weight)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    edgelist_path = Path(args.export_edgelist) if args.export_edgelist else None

    mat, node_names = load_matrix(in_path)
    data = matrix_to_graph(mat, keep_percent=args.keep_percent, keep_self_loops=args.self_loops, node_names=node_names)

    # Save the torch_geometric.data.Data object
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    if edgelist_path is not None:
        edgelist_path.parent.mkdir(parents=True, exist_ok=True)
        save_edgelist_csv(edgelist_path, data.edge_index, data.edge_attr)

    print(f"Saved graph to {out_path}")
    # print(f"Node names: {data.node_names}")
    if edgelist_path is not None:
        print(f"Exported edgelist CSV to {edgelist_path}")

if __name__ == "__main__":
    main()
