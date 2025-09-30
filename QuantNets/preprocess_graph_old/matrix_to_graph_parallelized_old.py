import os
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle
from sklearn.preprocessing import StandardScaler

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

def matrix_to_graph(mat: np.ndarray, keep_percent: float = 0.5, keep_self_loops: bool = False, 
                   node_names: list[str] = None, edge_stats_dict=None) -> Data:
    """Convert matrix to PyTorch Geometric graph with optional per-edge standardization."""
    # Basic checks
    if not (0 < keep_percent <= 1):
        raise ValueError("keep_percent must be in (0, 1]")
    n = mat.shape[0]

    # Make a copy of the matrix
    mat = np.array(mat, dtype=float)
    
    # Remove self-loops for edge selection
    np.fill_diagonal(mat, 0.0)

    # Work on upper triangle indices (i < j) - USING UNSTANDARDIZED VALUES
    iu, ju = np.triu_indices(n, k=1)
    vals = mat[iu, ju]

    # Handle NaNs by treating them as zero magnitude (won't be selected if keep_percent < 1)
    vals = np.nan_to_num(vals, nan=0.0)

    # Determine number of edges to keep
    total_upper = vals.size
    k = int(np.floor(keep_percent * total_upper))
    k = max(1, min(k, total_upper))

    # Get indices of top-k by absolute magnitude - BASED ON UNSTANDARDIZED VALUES
    order = np.argsort(np.abs(vals))[::-1]  # descending by |val|
    top_idx = order[:k]

    # Selected undirected pairs
    src_u = iu[top_idx]
    dst_u = ju[top_idx]
    
    # NOW apply PER-EDGE standardization to the SELECTED edges only
    if edge_stats_dict is not None:
        w_u_unstandardized = vals[top_idx]
        w_u_standardized = []
        
        for idx, (i, j, val) in enumerate(zip(src_u, dst_u, w_u_unstandardized)):
            # Create edge key (ensure consistent ordering: smaller index first)
            edge_key = (min(i, j), max(i, j))
            
            mean_ij = edge_stats_dict[edge_key]['mean']
            std_ij = edge_stats_dict[edge_key]['std']
            
            # Standardize using edge-specific statistics
            standardized_val = (val - mean_ij) / std_ij
                
            w_u_standardized.append(standardized_val)
                
        w_u = np.array(w_u_standardized)
    else:
        w_u = vals[top_idx]

    # Build edge_index with both directions to represent an undirected graph in PyG
    src = np.concatenate([src_u, dst_u], axis=0)
    dst = np.concatenate([dst_u, src_u], axis=0)
    w = np.concatenate([w_u, w_u], axis=0)

    # Use torch.from_numpy for zero-copy conversion
    edge_index_np = np.array([src, dst])
    edge_index = torch.from_numpy(edge_index_np).long()
    edge_attr = torch.from_numpy(w).float()

    # Node indices (brain region identity)
    x = torch.arange(n, dtype=torch.int64)
    
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

def collect_training_edge_statistics(train_csv_files: list[Path]) -> dict:
    """Collect per-edge statistics (mean, std) from training matrices."""
    print("Collecting per-edge statistics from training matrices...")
    
    # Dictionary to store edge values: {(i,j): [val1, val2, ...]}
    edge_values = {}
    
    for csv_file in train_csv_files:
        try:
            mat, _ = load_matrix(csv_file)
            n = mat.shape[0]
            
            # Get upper triangle indices (excluding diagonal)
            iu, ju = np.triu_indices(n, k=1)
            
            for i, j, val in zip(iu, ju, mat[iu, ju]):
                # Handle NaNs by treating them as zero
                if np.isnan(val):
                    val = 0.0
                
                # Create edge key (ensure consistent ordering)
                edge_key = (min(i, j), max(i, j))
                
                if edge_key not in edge_values:
                    edge_values[edge_key] = []
                edge_values[edge_key].append(val)
                
        except Exception as e:
            print(f"Error processing {csv_file.name} for edge statistics: {e}")
            continue
    
    if not edge_values:
        raise ValueError("No training edge values collected")
    
    # Compute statistics for each edge
    edge_stats = {}
    for edge_key, values in edge_values.items():
        values_array = np.array(values)
        edge_stats[edge_key] = {
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'count': len(values_array)
        }
    print(f"Computed per-edge statistics for {len(edge_stats)} unique edges")
    print(f"Each edge has statistics from {len(train_csv_files)} training matrices")
    
    # Print some example statistics
    example_edges = list(edge_stats.keys())[:5]
    for edge in example_edges:
        stats = edge_stats[edge]
        print(f"Edge {edge}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
    
    return edge_stats

def process_single_file(csv_file: Path, output_dir: Path, keep_percent: float = 0.5, 
                       keep_self_loops: bool = True, edge_stats_dict=None) -> tuple[bool, str, str]:
    """Process a single CSV file and convert to graph."""
    try:
        basename = csv_file.stem
        prefix = basename[:17]  # Take first 17 characters as prefix
        output_file = output_dir / f"{prefix}_fc_graph.pt"
        
        # Load matrix and convert to graph
        mat, node_names = load_matrix(csv_file)
        data = matrix_to_graph(mat, keep_percent=keep_percent, keep_self_loops=keep_self_loops, 
                              node_names=node_names, edge_stats_dict=edge_stats_dict)
        
        # Save the graph
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_file)
        
        return True, basename, f"Successfully converted: {basename}"
        
    except Exception as e:
        return False, basename, f"Failed to convert {basename}: {str(e)}"

def process_all_matrices_with_standardization(input_dir: str, output_dir: str, train_ids_file: str = None,
                                            keep_percent: float = 0.5, keep_self_loops: bool = True, 
                                            max_workers: int = None) -> None:
    """Process all CSV files with per-edge standardization based on training data."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Determine training files for edge standardization
    if train_ids_file and os.path.exists(train_ids_file):
        # Load training IDs from file
        with open(train_ids_file, 'r') as f:
            train_ids = set(line.strip() for line in f)
        
        train_csv_files = []
        for csv_file in csv_files:
            # Extract ID from filename (first 17 characters)
            file_id = csv_file.stem[:17]
            if file_id in train_ids:
                train_csv_files.append(csv_file)
        
        print(f"Using {len(train_csv_files)} training files for per-edge standardization")
    else:
        # Use all files for standardization (not ideal but fallback)
        print("Warning: No training IDs file provided. Using all files for per-edge standardization.")
        train_csv_files = csv_files
    
    # Collect per-edge statistics
    edge_stats_dict = collect_training_edge_statistics(train_csv_files)
    
    # Save the edge statistics for later use
    stats_path = output_path / "edge_statistics.pkl"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'wb') as f:
        pickle.dump(edge_stats_dict, f)
    print(f"Edge statistics saved to {stats_path}")
    
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
            executor.submit(process_single_file, csv_file, output_path, keep_percent, 
                          keep_self_loops, edge_stats_dict): csv_file 
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

def process_all_matrices(input_dir: str, output_dir: str, keep_percent: float = 0.5, 
                        keep_self_loops: bool = True, max_workers: int = None) -> None:
    """Process all CSV files in input directory using parallel processing (without standardization)."""
    
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
            executor.submit(process_single_file, csv_file, output_dir, keep_percent, keep_self_loops): csv_file 
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
    INPUT_DIR = "/scratch/bng/cartbind/data/FC_matrices/GF"
    OUTPUT_DIR = "/scratch/bng/cartbind/data/FC_graphs/raw/GF30"
    TRAIN_IDS_FILE = "/scratch/bng/cartbind/data/FC_graphs/processed/GF30/training_ids.txt"
    
    # Processing parameters
    KEEP_PERCENT = 0.3
    KEEP_SELF_LOOPS = True
    MAX_WORKERS = None  # Use all available CPUs, or set to specific number
    USE_STANDARDIZATION = True  # Set to True for per-edge standardization
    
    # Process all matrices
    if USE_STANDARDIZATION:
        process_all_matrices_with_standardization(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            train_ids_file=TRAIN_IDS_FILE,
            keep_percent=KEEP_PERCENT,
            keep_self_loops=KEEP_SELF_LOOPS,
            max_workers=MAX_WORKERS
        )
    else:
        process_all_matrices(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            keep_percent=KEEP_PERCENT,
            keep_self_loops=KEEP_SELF_LOOPS,
            max_workers=MAX_WORKERS
        )

if __name__ == "__main__":
    main()