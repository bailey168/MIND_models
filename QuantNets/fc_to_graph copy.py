
import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

def load_matrix(path: Path) -> np.ndarray:
    mat = np.loadtxt(path, delimiter=",")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be a square 2D matrix")
    
    # Check if matrix is symmetric
    if not np.allclose(mat, mat.T, rtol=1e-10, atol=1e-10):
        raise ValueError("Input matrix must be symmetric")

    return mat

def matrix_to_graph(mat: np.ndarray, keep_percent: float = 0.5, keep_self_loops: bool = False) -> Data:
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

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float32)

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
    # Save original (signed) weights as edge_attr and also store an absolute view if needed later
    data.abs_edge_attr = torch.abs(edge_attr)

    return data

def maybe_save_edgelist_csv(path: Path, edge_index: torch.Tensor, edge_attr: torch.Tensor):
    # For quick inspection/debugging
    rows = torch.stack([edge_index[0], edge_index[1], edge_attr], dim=1).cpu().numpy()
    header = "source,target,weight"
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt=["%d", "%d", "%.6f"])

def main():
    parser = argparse.ArgumentParser(description="Convert an FC matrix to a PyTorch Geometric graph.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to FC matrix (.npy or .csv)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .pt file to save the graph Data object")
    parser.add_argument("--keep-prop", "-p", type=float, default=0.5, help="Proportion of top-magnitude edges to keep from the upper triangle (0-1]")
    parser.add_argument("--self-loops", action="store_true", help="Include self-loops with weight 0.0")
    parser.add_argument("--export-edgelist", type=str, default="", help="Optional CSV path to export (source,target,weight)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    edgelist_path = Path(args.export_edgelist) if args.export_edgelist else None

    mat = load_matrix(in_path)
    data = matrix_to_graph(mat, keep_percent=args.keep_percent, keep_self_loops=args.self_loops)

    # Save the torch_geometric.data.Data object
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    if edgelist_path is not None:
        edgelist_path.parent.mkdir(parents=True, exist_ok=True)
        maybe_save_edgelist_csv(edgelist_path, data.edge_index, data.edge_attr)

    print(f"Saved graph to {out_path}")
    if edgelist_path is not None:
        print(f"Exported edgelist CSV to {edgelist_path}")

if __name__ == "__main__":
    main()
