import time
import os
import shutil
import pickle
import matplotlib
from matplotlib import pyplot as plt
import statistics
import torch
import numpy as np
from sklearn.metrics import r2_score

def time_it(func):
    """Decorator to time function execution."""
    def wrapper_function(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print(f'Function {func.__name__} took: {stop-start}s')
        return res
    return wrapper_function

def create_experiment_folder_structure(base_path, experiment_id):
    """Create folder structure for experiment results."""
    if not os.path.exists(base_path):
        print("Ensure that your base path exists -> {}".format(base_path))
        import sys
        sys.exit(1)
    
    experiments_dir = os.path.join(base_path, "Experiments_FC_09_18_gridsearch_layers")
    if not os.path.exists(experiments_dir):
        os.mkdir(experiments_dir)
    
    underscored_experiment_id = "_".join(str(experiment_id).strip().split(" "))
    specific_run_dir = os.path.join(experiments_dir, "run_" + underscored_experiment_id)
    if not os.path.exists(specific_run_dir):
        os.mkdir(specific_run_dir)
    
    # Create model-specific directories
    qgcn_dir = os.path.join(specific_run_dir, "qgcn")
    sgcn_dir = os.path.join(specific_run_dir, "sgcn")
    
    for dir_path in [qgcn_dir, sgcn_dir]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    
    return specific_run_dir, qgcn_dir, sgcn_dir

def copy_architecture_files(base_path, qgcn_dir, sgcn_dir):
    """Copy architecture files to experiment directories."""
    try:
        gnn_source_filepath = os.path.join(base_path, "gnn", "architectures.py")
        if os.path.exists(gnn_source_filepath):
            for dest_dir in [qgcn_dir, sgcn_dir]:
                dest_filepath = os.path.join(dest_dir, "architectures.py")
                if not os.path.exists(dest_filepath):
                    shutil.copyfile(gnn_source_filepath, dest_filepath)
    except Exception as e:
        print(f"Warning: Could not copy architecture files: {e}")

def move_graph_data_to_device(data, device):
    """Move graph data to target device."""
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.to(device)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)
    if hasattr(data, 'y') and data.y is not None:
        data.y = data.y.to(device)
    if hasattr(data, 'pos') and data.pos is not None:
        data.pos = data.pos.to(device)
    if hasattr(data, 'batch') and data.batch is not None:
        data.batch = data.batch.to(device)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    if hasattr(data, 'demographics') and data.demographics is not None:
        data.demographics = data.demographics.to(device)
    return data

def profile_model_complexity(model, data_sample, model_name, device, num_runs=10):
    """Profile a model's complexity and timing."""
    try:
        from flops_counter.ptflops import get_model_complexity_info
    except ImportError:
        print("ptflops not available for model profiling")
        return
    
    model.eval()
    
    try:
        macs, params = get_model_complexity_info(
            model, data_sample, as_strings=False, 
            print_per_layer_stat=False, verbose=False
        )
        flops = round(2*macs / 1e3, 3)
        macs = round(macs / 1e3, 3)
        params = round(params / 1e3, 3)
        
        # Profile inference wall time
        wall_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(data_sample.clone().detach().to(device))
            end_time = time.time()
            wall_times.append(end_time - start_time)
        
        wall_time_mean = statistics.mean(wall_times)
        wall_time_std = statistics.stdev(wall_times)
        
        print(f"\n-----------------")
        print(f"{model_name} Model Stats (Brain Connectivity):")
        print(f"Profiling data sample: {data_sample}")
        print("-------------------------------------------------------------------------------------------")
        print(f'Number of parameters: {params} k')
        print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
        print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
        print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
        print("-------------------------------------------------------------------------------------------")
        
    except Exception as e:
        print(f"Error profiling {model_name} model: {e}")

def evaluate_model_performance(model, dataloader, device, target_scaler_mean=None, target_scaler_std=None):
    """Evaluate a model and return MSE and R² in original scale."""
    model.eval()
    
    total_mse, total_samples = 0, 0
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for data in dataloader:
            data = move_graph_data_to_device(data, device)
            predictions = model(data)
            
            # Convert predictions and targets back to original scale
            if target_scaler_mean is not None and target_scaler_std is not None:
                predictions_orig = predictions.cpu().numpy() * target_scaler_std + target_scaler_mean
                targets_orig = data.y.cpu().numpy() * target_scaler_std + target_scaler_mean
            else:
                predictions_orig = predictions.cpu().numpy()
                targets_orig = data.y.cpu().numpy()
            
            # Calculate MSE in original scale
            mse = np.mean((predictions_orig - targets_orig) ** 2)
            total_mse += mse * len(predictions_orig)
            total_samples += len(predictions_orig)
            
            all_predictions.extend(predictions_orig)
            all_targets.extend(targets_orig)
    
    final_mse = total_mse / total_samples
    final_r2 = r2_score(all_targets, all_predictions) if len(all_predictions) > 1 else 0
    
    return final_mse, final_r2

def save_model_with_metadata(model, optimizer, save_dir, epoch_info=None, target_scaler_info=None):
    """Save model with metadata."""
    # Save complete model first (most reliable for loading)
    torch.save(model, os.path.join(save_dir, "model_complete.pth"))
    
    # Save model with metadata
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'model_type': type(model).__name__,
            'architecture': str(model)
        }
    }
    
    # Add epoch information if provided
    if epoch_info:
        save_dict.update(epoch_info)
    
    # Add target scaling information if provided
    if target_scaler_info:
        save_dict['target_scaling'] = target_scaler_info
    
    torch.save(save_dict, os.path.join(save_dir, "model.pth"))

def save_training_info(save_dir, final_epoch, model_type, early_stopping_config=None, early_stopping_obj=None):
    """Save training information to a text file."""
    with open(os.path.join(save_dir, "training_info.txt"), 'w') as f:
        f.write(f"Final training epoch: {final_epoch}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Early stopping: {'Yes' if early_stopping_config else 'No'}\n")
        if early_stopping_config and early_stopping_obj:
            f.write(f"Early stopped: {'Yes' if early_stopping_obj.early_stop else 'No'}\n")
            if hasattr(early_stopping_obj, 'best_epoch'):
                f.write(f"Best model epoch: {getattr(early_stopping_obj, 'best_epoch', 'unknown')}\n")
                f.write(f"Monitor metric: {early_stopping_config.get('monitor', 'loss')}\n")

def cache_training_results(results_dict, save_dirs):
    """Cache training results to pickle files."""
    for model_type, save_dir in save_dirs.items():
        if save_dir is None:
            continue
            
        # Save loss arrays
        if f'train_{model_type}_loss' in results_dict:
            filepath = os.path.join(save_dir, "train_loss.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'train_{model_type}_loss'], f)
        
        # Save MSE arrays
        if f'train_{model_type}_mse' in results_dict:
            filepath = os.path.join(save_dir, "train_mse.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'train_{model_type}_mse'], f)
        
        if f'test_{model_type}_mse' in results_dict:
            filepath = os.path.join(save_dir, "test_mse.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'test_{model_type}_mse'], f)
        
        # Save R² arrays
        if f'train_{model_type}_r2' in results_dict:
            filepath = os.path.join(save_dir, "train_r2.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'train_{model_type}_r2'], f)
        
        if f'test_{model_type}_r2' in results_dict:
            filepath = os.path.join(save_dir, "test_r2.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'test_{model_type}_r2'], f)
        
        # Save learning rates
        if f'learning_rates_{model_type}' in results_dict:
            filepath = os.path.join(save_dir, "learning_rates.pk")
            with open(filepath, 'wb') as f:
                pickle.dump(results_dict[f'learning_rates_{model_type}'], f)

def plot_training_history(data, labels):
    """Plot training history."""
    font = {'weight':'bold', 'size':8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.size':8})

    indicators = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    leftover_count = len(data) % 2
    num_rows = len(data) // 2
    col_count = 2
    row_count = num_rows + leftover_count
    fig, ax = plt.subplots(nrows=row_count, ncols=col_count, figsize=(12, 8))
    fig.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # Loop and plot the data and their labels
    for i, row_plts in enumerate(ax):
        for j, row_col_plt in enumerate(row_plts):
            data_index = i * col_count + j
            if data_index < len(data):
                xdata = list(range(1, len(data[data_index]) + 1))
                ydata = data[data_index]
                data_label = labels[data_index]
                data_indicator = indicators[data_index % len(indicators)]
                row_col_plt.plot(xdata, ydata, color=data_indicator, label=data_label)
                row_col_plt.set_xticks(xdata)
                row_col_plt.legend(loc="upper right")
                row_col_plt.set_xlabel('Epoch')
                row_col_plt.set_ylabel(data_label)
                row_col_plt.set_title('{} vs. No. of epochs'.format(data_label))
            else:
                row_col_plt.set_visible(False)
    plt.show()

class TargetScaler:
    """Handles target variable scaling."""
    
    def __init__(self, use_scaling=True):
        self.use_scaling = use_scaling
        self.mean = None
        self.std = None
    
    def fit_and_apply(self, data_struct, sgcn_exists, qgcn_exists):
        """Fit scaler and apply to datasets."""
        if not self.use_scaling:
            return
        
        print("Fitting target scaler on training data...")
        
        # Collect all training targets
        train_targets = []
        
        # Use SGCN training data (or QGCN if SGCN doesn't exist)
        if sgcn_exists:
            for data in data_struct["geometric"]["sgcn_train_data"]:
                train_targets.append(data.y.item() if data.y.dim() == 0 else data.y.cpu().numpy())
        elif qgcn_exists:
            for data in data_struct["geometric"]["qgcn_train_data"]:
                train_targets.append(data.y.item() if data.y.dim() == 0 else data.y.cpu().numpy())
        
        train_targets = np.array(train_targets)
        
        # Calculate mean and std
        self.mean = np.mean(train_targets)
        self.std = np.std(train_targets)
        
        print(f"Target scaling - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        
        # Apply scaling to all datasets
        self._apply_scaling(data_struct, sgcn_exists, qgcn_exists)
    
    def _apply_scaling(self, data_struct, sgcn_exists, qgcn_exists):
        """Apply target scaling to all datasets."""
        print("Applying target scaling to datasets...")
        
        # Scale SGCN datasets
        if sgcn_exists:
            for data in data_struct["geometric"]["sgcn_train_data"]:
                data.y = (data.y - self.mean) / self.std
            for data in data_struct["geometric"]["sgcn_test_data"]:
                data.y = (data.y - self.mean) / self.std
        
        # Scale QGCN datasets
        if qgcn_exists:
            for data in data_struct["geometric"]["qgcn_train_data"]:
                data.y = (data.y - self.mean) / self.std
            for data in data_struct["geometric"]["qgcn_test_data"]:
                data.y = (data.y - self.mean) / self.std
    
    def inverse_transform(self, scaled_targets):
        """Convert scaled targets back to original scale."""
        if self.use_scaling and self.mean is not None and self.std is not None:
            return scaled_targets * self.std + self.mean
        return scaled_targets
    
    def transform(self, original_targets):
        """Convert original targets to scaled targets."""
        if self.use_scaling and self.mean is not None and self.std is not None:
            return (original_targets - self.mean) / self.std
        return original_targets