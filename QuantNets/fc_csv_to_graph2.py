#!/usr/bin/env python3
"""
fc_csv_to_graph.py

Convert a symmetric functional-connectivity (FC) CSV matrix into a graph.

- Uses only the upper triangle (excluding the diagonal).
- Keeps the top X% of edges by absolute magnitude (default: 50%).
- Builds one-hot node features to preserve brain region identity.
- Saves a PyTorch Geometric Data object (.pt) with edge_index, edge_attr, x.
- Optionally writes an edgelist CSV for inspection.

Examples
--------
# Case 1: matrix with labels in CSV first column and header row
python fc_csv_to_graph.py \
  --csv fc_matrix.csv \
  --labels-in-csv \
  --keep-percent 50 \
  --out graph.pt

# Case 2: matrix without labels, provide a regions file (one name per line)
python fc_csv_to_graph.py \
  --csv fc_matrix.csv \
  --regions regions.txt \
  --keep-percent 40 \
  --out graph.pt \
  --edge-csv edges.csv
"""
import argparse
import math
import os
from typing import List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("This script requires pandas. Please install it (e.g., pip install pandas).") from e

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
except ImportError as e:
    raise SystemExit(
        "This script requires PyTorch and PyTorch Geometric.\n"
        "Install PyTorch per your platform, then install PyG, e.g.:\n"
        "  pip install torch\n"
        "  pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)')+cpu.html\n"
        "or follow https://pytorch-geometric.readthedocs.io"
    ) from e


def load_matrix_and_labels(csv_path: str, labels_in_csv: bool, regions_path: str | None) -> Tuple[np.ndarray, List[str]]:
    """
    Load the FC matrix and region labels.

    If labels_in_csv is True, the CSV is expected to have a header row and the
    first column as row labels. Otherwise, the CSV is treated as a pure numeric
    matrix. If regions_path is provided, its names override any CSV labels.

    Returns
    -------
    matrix : (N, N) np.ndarray of float
    labels : list[str] of length N
    """
    if labels_in_csv:
        df = pd.read_csv(csv_path, header=0, index_col=0)
        labels = df.index.astype(str).tolist()
        # If the columns also have names, ensure their order matches
        # and drop any duplicate label rows/cols if present.
        # Reindex columns to be in the same order as rows when possible.
        try:
            df = df.loc[labels, labels]
        except Exception:
            pass
        matrix = df.to_numpy(dtype=float)
    else:
        df = pd.read_csv(csv_path, header=None)
        matrix = df.to_numpy(dtype=float)
        labels = [f"ROI_{i}" for i in range(matrix.shape[0])]

    if regions_path is not None:
        with open(regions_path, "r", encoding="utf-8") as f:
            override = [line.strip() for line in f if line.strip() != ""]
        if len(override) != matrix.shape[0]:
            raise ValueError(
                f"regions file has {len(override)} names but matrix is {matrix.shape[0]}x{matrix.shape[0]}"
            )
        labels = override

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {matrix.shape}")

    return matrix, labels


def ensure_symmetric(matrix: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """Validate near-symmetry; if not, symmetrize by (A + A.T)/2 and warn."""
    if not np.allclose(matrix, matrix.T, atol=atol, equal_nan=False):
        # print a small warning but continue after symmetrization
        print("[WARN] Matrix not perfectly symmetric; symmetrizing by (A + A.T)/2.")
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def top_k_upper_triangle_edges(matrix: np.ndarray, keep_percent: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From the upper triangle (k=1 excludes diagonal), select the top-k edges by |weight|.

    Returns
    -------
    src : (k,) np.ndarray of int
    dst : (k,) np.ndarray of int
    w   : (k,) np.ndarray of float (original signed weights)
    """
    n = matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = matrix[iu, ju]

    total_edges = vals.size
    if total_edges == 0:
        raise ValueError("Matrix too small to form edges.")

    k = int(math.ceil((keep_percent / 100.0) * total_edges))
    k = max(1, min(k, total_edges))

    # Select indices of the k largest |vals| without full sort
    idx = np.argpartition(np.abs(vals), -k)[-k:]
    # For reproducible ordering (optional), sort the selected by descending |vals|
    idx = idx[np.argsort(-np.abs(vals[idx]))]

    src = iu[idx]
    dst = ju[idx]
    w = vals[idx]
    return src, dst, w


def build_pyg_graph(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, labels: List[str]) -> Data:
    """
    Build a PyG Data object with:
      - x: one-hot node features (N x N)
      - edge_index: undirected edges
      - edge_attr: weights aligned with edge_index
      - region_names: list[str] as an attribute
    """
    # Create directed edges (upper-tri only), then convert to undirected with duplicated weights
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float)

    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr, num_nodes=n)

    x = torch.eye(n, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)
    # Stash useful metadata
    data.region_names = labels  # list[str]
    return data


def write_edgelist_csv(path: str, src: np.ndarray, dst: np.ndarray, w: np.ndarray, labels: List[str]) -> None:
    """Write the selected (upper-tri) edges as a CSV with human-readable region names."""
    out = pd.DataFrame({
        "u": src,
        "v": dst,
        "u_label": [labels[i] for i in src],
        "v_label": [labels[j] for j in dst],
        "weight": w,
        "abs_weight": np.abs(w),
    })
    out.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Convert FC CSV matrix to a PyTorch Geometric graph.")
    parser.add_argument("--csv", required=True, help="Path to symmetric FC matrix CSV.")
    parser.add_argument("--out", required=False, default=None, help="Output .pt file to save the PyG Data object.")
    parser.add_argument("--keep-percent", type=float, default=50.0, help="Percent of upper-tri edges to keep by |weight| (0-100].")
    parser.add_argument("--labels-in-csv", action="store_true", help="Treat first column as row labels and header row as column labels.")
    parser.add_argument("--regions", default=None, help="Optional path to a text file of region names (one per line). Overrides CSV labels if provided.")
    parser.add_argument("--edge-csv", default=None, help="Optional path to write the selected (upper-tri) edges as a CSV.")
    args = parser.parse_args()

    if not (0 < args.keep_percent <= 100):
        raise SystemExit("--keep-percent must be in (0, 100].")

    matrix, labels = load_matrix_and_labels(args.csv, args.labels_in_csv, args.regions)
    matrix = ensure_symmetric(matrix)
    np.fill_diagonal(matrix, 0.0)  # Ignore self-connections

    n = matrix.shape[0]
    src_ut, dst_ut, w_ut = top_k_upper_triangle_edges(matrix, args.keep_percent)

    # Stats
    total_ut = n * (n - 1) // 2
    kept_ut = src_ut.size
    print(f"[INFO] Nodes: {n}")
    print(f"[INFO] Upper-tri edges available: {total_ut}")
    print(f"[INFO] Keeping top {args.keep_percent:.2f}% => {kept_ut} edges (upper-tri).")

    data = build_pyg_graph(n, src_ut, dst_ut, w_ut, labels)

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = f"{base}_top{int(args.keep_percent)}.pt"
    torch.save(data, out_path)
    print(f"[OK] Saved PyG Data to: {out_path}")

    if args.edge_csv:
        write_edgelist_csv(args.edge_csv, src_ut, dst_ut, w_ut, labels)
        print(f"[OK] Wrote edge CSV to: {args.edge_csv}")

    # Also save region names next to the graph for convenience
    regions_out = os.path.splitext(out_path)[0] + ".regions.txt"
    with open(regions_out, "w", encoding="utf-8") as f:
        for name in labels:
            f.write(f"{name}\n")
    print(f"[OK] Wrote region names to: {regions_out}")


if __name__ == "__main__":
    main()
