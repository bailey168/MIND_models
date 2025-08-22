import os
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def load_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load matrix from CSV file with row labels."""
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
    """Convert matrix to PyTorch Geometric graph."""
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

def process_single_file(csv_file: Path, output_dir: Path, keep_percent: float = 0.5, keep_self_loops: bool = True) -> tuple[bool, str, str]:
    """Process a single CSV file and convert to graph."""
    try:
        basename = csv_file.stem
        prefix = basename[:17]  # Take first 17 characters as prefix
        output_file = output_dir / f"{prefix}_fc_graph.pt"
        
        # Load matrix and convert to graph
        mat, node_names = load_matrix(csv_file)
        data = matrix_to_graph(mat, keep_percent=keep_percent, keep_self_loops=keep_self_loops, node_names=node_names)
        
        # Save the graph
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_file)
        
        return True, basename, f"Successfully converted: {basename}"
        
    except Exception as e:
        return False, basename, f"Failed to convert {basename}: {str(e)}"

def process_all_matrices(input_dir: str, output_dir: str, keep_percent: float = 0.5, 
                        keep_self_loops: bool = True, max_workers: int = None) -> None:
    """Process all CSV files in input directory using parallel processing."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(csv_files))
    
    print(f"Using {max_workers} workers for parallel processing")
    
    start_time = time.time()
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, csv_file, output_path, keep_percent, keep_self_loops): csv_file 
            for csv_file in csv_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            csv_file = future_to_file[future]
            try:
                success, basename, message = future.result()
                print(message)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Unexpected error processing {csv_file.name}: {str(e)}")
                failed += 1
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Total runtime: {runtime:.2f} seconds")

def main():
    """Main function to process FC matrices."""
    
    # Configuration - modify these paths as needed
    INPUT_DIR = "/scratch/bng/cartbind/data/FC_matrices/TMT"
    OUTPUT_DIR = "/scratch/bng/cartbind/data/FC_graphs/raw/TMT"
    
    # Processing parameters
    KEEP_PERCENT = 0.5
    KEEP_SELF_LOOPS = True
    MAX_WORKERS = None  # Use all available CPUs, or set to specific number
    
    # Process all matrices
    process_all_matrices(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        keep_percent=KEEP_PERCENT,
        keep_self_loops=KEEP_SELF_LOOPS,
        max_workers=MAX_WORKERS
    )

if __name__ == "__main__":
    main()