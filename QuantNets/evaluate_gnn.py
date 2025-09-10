import os
import sys
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from util.data_processing import read_cached_graph_dataset
from torch_geometric.loader import DataLoader as GraphDataLoader

class ModelEvaluator:
    def __init__(self, model_path, dataset_config, base_path="."):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the saved model (.pth file)
            dataset_config: Dataset configuration dict
            base_path: Base path for data files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_path = base_path
        self.dataset_config = dataset_config
        
        # Load the model using the robust loading method
        self.model = self._load_model_robust(model_path)
        self.model.eval()
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load dataset
        self.data_struct = read_cached_graph_dataset(
            num_train=dataset_config.get('num_train'),
            num_test=dataset_config.get('num_test'),
            dataset_name=dataset_config['dataset_name'],
            parent_dir=base_path
        )
    
    def _load_model_robust(self, model_path):
        """
        Robustly load a model, trying multiple approaches.
        
        Args:
            model_path: Path to the saved model (.pth file)
            
        Returns:
            Loaded model
        """
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, "model_metadata.pth")
        
        # Method 1: Try loading with metadata (most robust)
        if os.path.exists(metadata_path):
            try:
                print("Attempting to load model using metadata...")
                metadata = torch.load(metadata_path, map_location=self.device, weights_only=False)
                
                # Import model classes
                from gnn.architectures import GraphConvNet, GATv2ConvNet
                
                model_class_name = metadata['model_class']
                model_config = metadata['model_config']
                
                # Create model based on class name and config
                if model_class_name == 'GraphConvNet':
                    model = GraphConvNet(
                        out_dim=model_config.get('out_dim', 1),
                        input_features=21,  # From your config
                        output_channels=32,  # From your config  
                        layers_num=model_config.get('layers_num', 3),
                        model_dim=32,  # From your config
                        embedding_dim=16,  # From your config
                        include_demo=model_config.get('include_demo', True),
                        demo_dim=model_config.get('demo_dim', 4)
                    )
                elif model_class_name == 'GATv2ConvNet':
                    model = GATv2ConvNet(
                        out_dim=model_config.get('out_dim', 1),
                        input_features=21,  # From your config
                        output_channels=32,  # From your config
                        layers_num=model_config.get('layers_num', 3),
                        model_dim=32,  # From your config
                        embedding_dim=16,  # From your config
                        include_demo=model_config.get('include_demo', True),
                        demo_dim=model_config.get('demo_dim', 4)
                    )
                else:
                    raise ValueError(f"Unknown model class: {model_class_name}")
                
                # Load the state dict
                model.load_state_dict(metadata['model_state_dict'])
                model.to(self.device)
                print(f"Successfully loaded {model_class_name} using metadata")
                return model
                
            except Exception as e:
                print(f"Failed to load using metadata: {e}")
                print("Falling back to direct model loading...")
        
        # Method 2: Try loading the complete model directly (backward compatibility)
        try:
            print("Attempting to load complete model object...")
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.to(self.device)
            print("Successfully loaded complete model object")
            return model
            
        except Exception as e:
            print(f"Failed to load complete model: {e}")
            
        # Method 3: Try loading with weights_only=True as a last resort
        try:
            print("Attempting to load with weights_only=True...")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # This requires manual model creation - would need more config info
            print("Loaded state dict, but need model architecture info to proceed")
            raise RuntimeError("Cannot create model from state dict alone - need architecture information")
            
        except Exception as e:
            raise RuntimeError(f"All model loading methods failed. Error: {e}")
    
    def _move_data_to_device(self, data):
        """Move graph data to target device."""
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.to(self.device)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index = data.edge_index.to(self.device)
        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(self.device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(self.device)
        if hasattr(data, 'batch') and data.batch is not None:
            data.batch = data.batch.to(self.device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(self.device)
        if hasattr(data, 'demographics') and data.demographics is not None:
            data.demographics = data.demographics.to(self.device)
        return data
    
    def evaluate_dataset(self, dataset_type='test', batch_size=64):
        """
        Evaluate the model on train or test dataset.
        
        Args:
            dataset_type: 'train' or 'test'
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Select the appropriate dataset
        if dataset_type == 'train':
            dataset = self.data_struct["geometric"]["sgcn_train_data"]
        elif dataset_type == 'test':
            dataset = self.data_struct["geometric"]["sgcn_test_data"]
        else:
            raise ValueError("dataset_type must be 'train' or 'test'")
        
        # Create data loader
        data_loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Collect predictions and true values
        all_predictions = []
        all_true_values = []
        
        print(f"Evaluating on {dataset_type} dataset...")
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data = self._move_data_to_device(data)
                
                # Get predictions
                predictions = self.model(data)
                
                # Handle different output formats
                if isinstance(predictions, torch.Tensor):
                    if predictions.dim() == 0:  # Scalar output
                        predictions = predictions.unsqueeze(0)
                    elif predictions.dim() == 2 and predictions.size(1) == 1:  # (batch_size, 1)
                        predictions = predictions.squeeze(-1)
                
                # Collect predictions and true values
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(data.y.cpu().numpy())
                
                # if (batch_idx + 1) % 10 == 0:
                #     print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Concatenate all results
        predictions = np.concatenate(all_predictions)
        true_values = np.concatenate(all_true_values)
        
        # Calculate metrics
        r2 = r2_score(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        
        # Calculate additional metrics
        correlation = np.corrcoef(true_values, predictions)[0, 1]
        
        results = {
            'dataset_type': dataset_type,
            'n_samples': len(predictions),
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'predictions': predictions,
            'true_values': true_values
        }
        
        return results
    
    def plot_predictions(self, results, save_path=None, show_plot=False):
        """
        Plot predictions vs true values.
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Optional path to save the plot
            show_plot: Whether to show the plot (requires interactive backend)
        """
        predictions = results['predictions']
        true_values = results['true_values']
        r2 = results['r2_score']
        dataset_type = results['dataset_type']
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(true_values, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Predictions vs True Values ({dataset_type.title()} Set)\nR² = {r2:.4f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make plot square
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def plot_training_curves(self, experiment_dir, save_path=None, show_plot=False):
        """
        Plot training and test MSE curves from saved pickle files.
        
        Args:
            experiment_dir: Directory containing train_mse.pk and test_mse.pk files
            save_path: Optional path to save the plot
            show_plot: Whether to show the plot (requires interactive backend)
        """
        train_mse_path = os.path.join(experiment_dir, "train_mse.pk")
        test_mse_path = os.path.join(experiment_dir, "test_mse.pk")
        
        # Check if files exist
        if not os.path.exists(train_mse_path):
            print(f"Train MSE file not found: {train_mse_path}")
            return
        if not os.path.exists(test_mse_path):
            print(f"Test MSE file not found: {test_mse_path}")
            return
        
        # Load the MSE data
        with open(train_mse_path, 'rb') as f:
            train_mse = pickle.load(f)
        with open(test_mse_path, 'rb') as f:
            test_mse = pickle.load(f)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(train_mse) + 1)
        
        plt.plot(epochs, train_mse, 'b-', linewidth=2, label='Training MSE', alpha=0.8)
        plt.plot(epochs, test_mse, 'r-', linewidth=2, label='Test MSE', alpha=0.8)
        
        # Add markers for better visibility
        plt.plot(epochs[::max(1, len(epochs)//20)], 
                [train_mse[i] for i in range(0, len(train_mse), max(1, len(epochs)//20))], 
                'bo', markersize=4, alpha=0.7)
        plt.plot(epochs[::max(1, len(epochs)//20)], 
                [test_mse[i] for i in range(0, len(test_mse), max(1, len(epochs)//20))], 
                'ro', markersize=4, alpha=0.7)
        
        # Formatting
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
        plt.title('Training and Test MSE Over Time', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        min_train_mse = min(train_mse)
        min_test_mse = min(test_mse)
        final_train_mse = train_mse[-1]
        final_test_mse = test_mse[-1]
        
        # Add text box with statistics
        stats_text = f'Final Train MSE: {final_train_mse:.6f}\n'
        stats_text += f'Final Test MSE: {final_test_mse:.6f}\n'
        stats_text += f'Best Train MSE: {min_train_mse:.6f}\n'
        stats_text += f'Best Test MSE: {min_test_mse:.6f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
        
        # Print summary
        print(f"\nTraining Curves Summary:")
        print(f"{'='*50}")
        print(f"Total epochs: {len(train_mse)}")
        print(f"Final Training MSE: {final_train_mse:.6f}")
        print(f"Final Test MSE: {final_test_mse:.6f}")
        print(f"Best Training MSE: {min_train_mse:.6f} (epoch {train_mse.index(min_train_mse) + 1})")
        print(f"Best Test MSE: {min_test_mse:.6f} (epoch {test_mse.index(min_test_mse) + 1})")
        print(f"{'='*50}")

    def print_evaluation_summary(self, results):
        """
        Print a summary of evaluation results.
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        dataset_type = results['dataset_type']
        n_samples = results['n_samples']
        r2 = results['r2_score']
        mse = results['mse']
        rmse = results['rmse']
        mae = results['mae']
        correlation = results['correlation']
        
        print(f"\n{'='*60}")
        print(f"{dataset_type.upper()} SET EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Number of samples: {n_samples}")
        print(f"R² Score: {r2:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Pearson Correlation: {correlation:.6f}")
        print(f"{'='*60}")


def load_config_from_experiment(experiment_dir):
    """
    Load configuration from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., "Experiments_FC/run_DSST_regression_lr_0.0001/sgcn")
    
    Returns:
        Configuration dictionary
    """
    # Try to load from saved experiment config first
    config_path = os.path.join(experiment_dir, "experiment_config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                exp_config = yaml.safe_load(f)
            dataset_config = exp_config.get('dataset_config', {})
            split_config = exp_config.get('split_config', {})
            
            return {
                'dataset_name': dataset_config.get('dataset_name', 'custom_dataset_selfloops_True_edgeft_None_norm_True'),
                'num_train': split_config.get('train', 27181),
                'num_test': split_config.get('test', 6796)
            }
        except Exception as e:
            print(f"Warning: Could not load experiment config: {e}")
    
    # Fallback to hardcoded config
    config = {
        'dataset_name': 'custom_dataset_selfloops_True_edgeft_None_norm_True',
        'num_train': 27181,
        'num_test': 6796
    }
    return config


def main():
    """Main evaluation function."""
    # Configuration - Update these paths according to your setup
    base_path = "/Users/baileyng/MIND_models/QuantNets"
    
    # Example: evaluate SGCN model
    experiment_id = "GF_regression_GATv2_config_1_lr_0.0001_epochs_496_scheduler_step"  # Update this to match your experiment
    model_path = os.path.join(base_path, "Experiments_FC_change_arch", f"run_{experiment_id}", "sgcn", "model.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available experiments:")
        experiments_dir = os.path.join(base_path, "Experiments_FC_change_arch")
        if os.path.exists(experiments_dir):
            for exp in os.listdir(experiments_dir):
                exp_path = os.path.join(experiments_dir, exp)
                if os.path.isdir(exp_path):
                    print(f"  - {exp}")
                    # Check for model files
                    sgcn_model = os.path.join(exp_path, "sgcn", "model.pth")
                    qgcn_model = os.path.join(exp_path, "qgcn", "model.pth")
                    if os.path.exists(sgcn_model):
                        print(f"    SGCN model: ✓")
                    if os.path.exists(qgcn_model):
                        print(f"    QGCN model: ✓")
        return
    
    # Load dataset configuration
    dataset_config = load_config_from_experiment(os.path.dirname(model_path))
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path, dataset_config, base_path)
        
        # Evaluate on test set
        test_results = evaluator.evaluate_dataset('test', batch_size=64)
        evaluator.print_evaluation_summary(test_results)
        
        # Evaluate on training set for comparison
        train_results = evaluator.evaluate_dataset('train', batch_size=64)
        evaluator.print_evaluation_summary(train_results)
        
        # Plot results
        plot_dir = os.path.dirname(model_path)
        evaluator.plot_predictions(test_results, save_path=os.path.join(plot_dir, "test_predictions.png"))
        evaluator.plot_predictions(train_results, save_path=os.path.join(plot_dir, "train_predictions.png"))
        
        # Plot training curves from saved MSE files
        evaluator.plot_training_curves(plot_dir, save_path=os.path.join(plot_dir, "training_curves.png"))
        
        # Save detailed results
        results_path = os.path.join(plot_dir, "evaluation_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({'train': train_results, 'test': test_results}, f)
        print(f"Detailed results saved to {results_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()