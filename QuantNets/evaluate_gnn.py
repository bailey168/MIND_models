import os
import sys
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
        
        # Load the model
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
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
    
    def plot_predictions(self, results, save_path=None):
        """
        Plot predictions vs true values.
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Optional path to save the plot
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
        
        plt.show()
    
    def print_evaluation_summary(self, results):
        """Print a summary of evaluation results."""
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY - {results['dataset_type'].upper()} SET")
        print(f"{'='*50}")
        print(f"Number of samples: {results['n_samples']}")
        print(f"R² Score: {results['r2_score']:.6f}")
        print(f"MSE: {results['mse']:.6f}")
        print(f"RMSE: {results['rmse']:.6f}")
        print(f"MAE: {results['mae']:.6f}")
        print(f"Correlation: {results['correlation']:.6f}")
        print(f"{'='*50}")


def load_config_from_experiment(experiment_dir):
    """
    Load configuration from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., "Experiments_FC/run_DSST_regression_lr_0.0001/sgcn")
    
    Returns:
        Configuration dictionary
    """
    # This is a basic config based on your files
    # You might want to save the actual config during training for better reproducibility
    # config = {
    #     'dataset_name': 'custom_dataset_selfloops_True_edgeft_None_norm_True',
    #     'num_train': 19420,
    #     'num_test': 4855
    # }
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
    experiment_id = "GF_regression_lr_1e-05"  # Update this to match your experiment
    model_path = os.path.join(base_path, "Experiments_FC", f"run_{experiment_id}", "sgcn", "model.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available experiments:")
        experiments_dir = os.path.join(base_path, "Experiments_FC")
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