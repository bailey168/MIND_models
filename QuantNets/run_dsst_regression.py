import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from util.reproducibility import set_deterministic_training
from itertools import product

# Set seed for reproducibility
SEED = 42
generator = set_deterministic_training(SEED)

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from gnn.architectures import GraphConvNet, GATv2ConvNet
from experiment_regression_DSST import ExperimentRegression

TARGET = 'GF'

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_model(model_type, dataset_config):
    """Create model based on type specification."""
    if model_type == 'GraphConv':
        return GraphConvNet(
            out_dim=dataset_config['out_dim'],
            input_features=dataset_config['in_channels'],
            output_channels=dataset_config['out_channels'],
            layers_num=dataset_config['layers_num'],
            model_dim=dataset_config['hidden_channels'],
            hidden_sf=dataset_config['graph_hidden_sf'],
            out_sf=dataset_config['graph_out_sf'],
            embedding_dim=dataset_config['embedding_dim'],
            include_demo=dataset_config['include_demo'],
            demo_dim=dataset_config['demo_dim'],
            dropout_rate=dataset_config['dropout_rate']
        )
    elif model_type == 'GATv2':
        return GATv2ConvNet(
            out_dim=dataset_config['out_dim'],
            input_features=dataset_config['in_channels'],
            output_channels=dataset_config['out_channels'],
            layers_num=dataset_config['layers_num'],
            model_dim=dataset_config['hidden_channels'],
            hidden_sf=dataset_config['gatv2_hidden_sf'],
            out_sf=dataset_config['gatv2_out_sf'],
            hidden_heads=dataset_config['gatv2_hidden_heads'],
            embedding_dim=dataset_config['embedding_dim'],
            include_demo=dataset_config['include_demo'],
            demo_dim=dataset_config['demo_dim'],
            dropout_rate=dataset_config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def generate_grid_search_configs(run_config, splits_config, datasets_config):
    """Generate all combinations of grid search parameters."""
    # Get grid search parameters
    grid_params = run_config.get('grid_search', {}).get('gf_custom', {})
    dropout_rates = grid_params['dropout_rates']
    weight_decays = grid_params['weight_decays']
    layers_nums = grid_params['layers_nums']

    # Get base configurations
    base_lr = run_config['lrs']['gf_custom'][0]
    base_epochs = run_config['epochs']['gf_custom'][0]
    base_scheduler = run_config.get('schedulers', {}).get('gf_custom', [{}])[0]
    base_early_stopping = run_config.get('early_stopping', {}).get('gf_custom', [{}])[0]
    base_split = splits_config['gf_custom']
    base_dataset = datasets_config['gf_custom']
    
    configs = []
    config_idx = 0
    
    # Generate all combinations - now including layers_num
    for dropout_rate, weight_decay, layers_num in product(dropout_rates, weight_decays, layers_nums):
        # Create modified dataset config
        dataset_config = base_dataset.copy()
        dataset_config['dropout_rate'] = dropout_rate
        dataset_config['weight_decay'] = weight_decay
        dataset_config['layers_num'] = layers_num  # Add layers_num to dataset config
        
        # Create experiment configuration
        config = {
            'config_idx': config_idx,
            'dataset_config': dataset_config,
            'split_config': base_split,
            'lr': base_lr,
            'epochs': base_epochs,
            'scheduler_config': base_scheduler,
            'early_stopping_config': base_early_stopping,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'layers_num': layers_num  # Add layers_num to config
        }
        
        configs.append(config)
        config_idx += 1
    
    return configs

def run_single_experiment(config, model_type, run_evaluation=True, use_target_scaling=True):
    """Run a single experiment with given configuration."""
    
    config_idx = config['config_idx']
    dataset_config = config['dataset_config']
    split_config = config['split_config']
    run_settings = config['lr']
    epochs = config['epochs']
    scheduler_config = config['scheduler_config']
    early_stopping_config = config['early_stopping_config']
    dropout_rate = config['dropout_rate']
    weight_decay = config['weight_decay']
    layers_num = config['layers_num']  # Extract layers_num
    
    print(f"\n{'='*80}")
    print(f"Starting Grid Search Experiment {config_idx + 1}")
    print(f"Model Type: {model_type}")
    print(f"Learning Rate: {run_settings}")
    print(f"Epochs: {epochs}")
    print(f"Dropout Rate: {dropout_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Layers Number: {layers_num}")  # Add layers_num to output
    print(f"Train/Test Split: {split_config['train']}/{split_config['test']}")
    print(f"Scheduler: {scheduler_config.get('scheduler', 'none')}")
    print(f"Early Stopping: {'Enabled' if early_stopping_config and early_stopping_config.get('enabled', False) else 'Disabled'}")
    print(f"Run Evaluation: {run_evaluation}")
    print(f"{'='*80}\n")
    
    # Create model
    model = create_model(model_type, dataset_config)
    
    # Prepare optimizer parameters with scheduler and weight_decay
    optim_params = {"lr": run_settings, "weight_decay": weight_decay}
    optim_params.update(scheduler_config)
    
    # Create unique experiment ID - now including layers_num
    early_stop_suffix = "_ES" if early_stopping_config and early_stopping_config.get('enabled', False) else ""
    experiment_id = f"{TARGET}_regression_{model_type}_grid_{config_idx + 1}_lr_{run_settings}_epochs_{epochs}_drop_{dropout_rate}_wd_{weight_decay}_layers_{layers_num}_scheduler_{scheduler_config.get('scheduler', 'none')}{early_stop_suffix}"
    
    # Setup experiment with target scaling
    experiment = ExperimentRegression(
        sgcn_model=model,
        qgcn_model=None,
        cnn_model=None,
        optim_params=optim_params,
        base_path=current_dir,
        num_train=split_config['train'],
        num_test=split_config['test'],
        dataset_name=dataset_config['dataset_name'],
        train_batch_size=split_config['batch_size'],
        test_batch_size=split_config['batch_size'],
        train_shuffle_data=True,
        test_shuffle_data=False,
        profile_run=False,
        id=experiment_id,
        early_stopping_config=early_stopping_config,
        use_target_scaling=use_target_scaling  # Add this parameter
    )
    
    # Run experiment with evaluation
    results = experiment.run(num_epochs=epochs, eval_training_set=True, run_evaluation=run_evaluation)
    
    # Calculate best test MSE
    best_test_mse = min(results['test_sgcn_mse']) if results['test_sgcn_mse'] else float('inf')
    best_test_mse_epoch = results['test_sgcn_mse'].index(best_test_mse) + 1 if results['test_sgcn_mse'] else 0
    
    # Create a summary file with detailed information including final epoch
    if hasattr(experiment, 'sgcn_specific_run_dir') and experiment.sgcn_specific_run_dir:
        # Save experiment configuration for evaluation script
        experiment_config = {
            'dataset_config': dataset_config,
            'split_config': split_config,
            'model_type': model_type,
            'experiment_id': experiment_id,
            'run_settings': run_settings,
            'planned_epochs': epochs,
            'actual_final_epoch': results.get('final_sgcn_epoch', epochs),
            'early_stopped': results.get('early_stopped', False),
            'scheduler_config': scheduler_config,
            'early_stopping_config': early_stopping_config,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'layers_num': layers_num  # Add layers_num to saved config
        }
        config_path = os.path.join(experiment.sgcn_specific_run_dir, "experiment_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f)
        
        # Create a human-readable summary
        summary_path = os.path.join(experiment.sgcn_specific_run_dir, "experiment_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Grid Search Experiment Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Learning Rate: {run_settings}\n")
            f.write(f"Dropout Rate: {dropout_rate}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write(f"Number of Layers: {layers_num}\n")  # Add layers_num to summary
            f.write(f"Planned Epochs: {epochs}\n")
            f.write(f"Final Training Epoch: {results.get('final_sgcn_epoch', epochs)}\n")
            f.write(f"Saved Model Epoch: {best_test_mse_epoch}\n")
            f.write(f"Early Stopped: {'Yes' if results.get('early_stopped', False) else 'No'}\n")
            f.write(f"Scheduler: {scheduler_config.get('scheduler', 'none')}\n")
            f.write(f"Final Test MSE: {results['test_sgcn_mse'][-1]:.5f}\n")
            f.write(f"Best Test MSE: {best_test_mse:.5f} (epoch {best_test_mse_epoch})\n")
            
            if 'evaluation' in results and results['evaluation']:
                eval_results = results['evaluation']
                f.write(f"Test R² Score: {eval_results['test_results']['r2_score']:.4f}\n")
                f.write(f"Test RMSE: {eval_results['test_results']['rmse']:.4f}\n")
                f.write(f"Train R² Score: {eval_results['train_results']['r2_score']:.4f}\n")
    
    print(f"\nGrid Search Experiment {config_idx + 1} completed!")
    print(f"Dropout: {dropout_rate}, Weight Decay: {weight_decay}, Layers: {layers_num}")  # Add layers_num to output
    print(f"Planned epochs: {epochs}, Final training epoch: {results.get('final_sgcn_epoch', epochs)}")
    print(f"Saved model from epoch: {best_test_mse_epoch}")
    print(f"Early stopped: {'Yes' if results.get('early_stopped', False) else 'No'}")
    print(f"Final Test MSE: SGCN={results['test_sgcn_mse'][-1]:.5f}")
    print(f"Best Test MSE: SGCN={best_test_mse:.5f} (epoch {best_test_mse_epoch})")
    
    # Print evaluation summary if available
    if 'evaluation' in results and results['evaluation']:
        eval_results = results['evaluation']
        print(f"\nEvaluation Summary for {eval_results['model_type']} model:")
        print(f"Test R² Score: {eval_results['test_results']['r2_score']:.4f}")
        print(f"Test RMSE: {eval_results['test_results']['rmse']:.4f}")
        print(f"Train R² Score: {eval_results['train_results']['r2_score']:.4f}")
        print(f"Plots saved to: {eval_results['model_path'].replace('model.pth', '')}")
    
    # Add grid search specific results
    results['grid_search_params'] = {
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay,
        'layers_num': layers_num,  # Add layers_num to results
        'config_idx': config_idx
    }
    
    return results

def run_grid_search(model_type, run_evaluation=True, use_target_scaling=True):
    """Run grid search experiments for all parameter combinations."""
    
    # Load configurations
    datasets_config = load_config(os.path.join(current_dir, 'datasets_DSST.yaml'))
    splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
    run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))
    
    # Generate all grid search configurations
    configs = generate_grid_search_configs(run_config, splits_config, datasets_config)
    
    print(f"Running Grid Search with {len(configs)} parameter combinations")
    print(f"Model: {model_type}")
    print(f"Post-training evaluation: {'Enabled' if run_evaluation else 'Disabled'}")
    
    # Print grid search overview
    grid_params = run_config.get('grid_search', {}).get('gf_custom', {})
    dropout_rates = grid_params['dropout_rates']
    weight_decays = grid_params['weight_decays']
    layers_nums = grid_params['layers_nums']

    print(f"\nGrid Search Parameters:")
    print(f"Dropout Rates: {dropout_rates}")
    print(f"Weight Decays: {weight_decays}")
    print(f"Layers Numbers: {layers_nums}")  # Add layers_nums to output
    print(f"Total Combinations: {len(dropout_rates)} × {len(weight_decays)} × {len(layers_nums)} = {len(configs)}")
    
    all_results = []
    
    # Run all experiments
    for config in configs:
        try:
            results = run_single_experiment(config, model_type, run_evaluation, use_target_scaling)
            
            # Calculate best test MSE for summary
            best_test_mse = min(results['test_sgcn_mse']) if results['test_sgcn_mse'] else float('inf')
            best_test_mse_epoch = results['test_sgcn_mse'].index(best_test_mse) + 1 if results['test_sgcn_mse'] else 0
            
            all_results.append({
                'config_idx': config['config_idx'],
                'model_type': model_type,
                'dropout_rate': config['dropout_rate'],
                'weight_decay': config['weight_decay'],
                'layers_num': config['layers_num'],  # Add layers_num to results
                'lr': config['lr'],
                'epochs': config['epochs'],
                'scheduler': config['scheduler_config'].get('scheduler', 'none'),
                'results': results,
                'best_test_mse': best_test_mse,
                'best_test_mse_epoch': best_test_mse_epoch
            })
            
        except Exception as e:
            print(f"Error in grid search experiment {config['config_idx'] + 1}: {str(e)}")
            print("Continuing with next experiment...")
            continue
    
    # Print summary of all experiments - update table header
    print(f"\n{'='*160}")
    print(f"GRID SEARCH SUMMARY - {model_type} MODEL")
    print(f"{'='*160}")
    print(f"{'Config':<8} {'Dropout':<10} {'WeightDec':<12} {'Layers':<8} {'LR':<10} {'Epochs':<8} {'Scheduler':<12} {'Final MSE':<12} {'Best MSE':<12} {'Test R²':<10}")
    print("-" * 160)
    
    for result in all_results:
        final_mse = result['results']['test_sgcn_mse'][-1] if result['results']['test_sgcn_mse'] else 'N/A'
        best_mse = result['best_test_mse'] if result['best_test_mse'] != float('inf') else 'N/A'
        test_r2 = 'N/A'
        if 'evaluation' in result['results'] and result['results']['evaluation']:
            test_r2 = f"{result['results']['evaluation']['test_results']['r2_score']:.4f}"
        
        print(f"{result['config_idx'] + 1:<8} {result['dropout_rate']:<10} {result['weight_decay']:<12} {result['layers_num']:<8} {result['lr']:<10} {result['epochs']:<8} {result['scheduler']:<12} {final_mse:<12.5f} {best_mse:<12.5f} {test_r2:<10}")
    
    # Find best performing model - update to include layers_num
    if all_results:
        valid_results = [r for r in all_results if r['results']['test_sgcn_mse']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['best_test_mse'])
            print(f"\nBest performing model:")
            print(f"Config {best_result['config_idx'] + 1}: Dropout={best_result['dropout_rate']}, WeightDecay={best_result['weight_decay']}, Layers={best_result['layers_num']}")
            print(f"Model={best_result['model_type']}, LR={best_result['lr']}, Epochs={best_result['epochs']}, Scheduler={best_result['scheduler']}")
            print(f"Best Test MSE: {best_result['best_test_mse']:.5f} (epoch {best_result['best_test_mse_epoch']})")
            print(f"Final Test MSE: {best_result['results']['test_sgcn_mse'][-1]:.5f}")
            if 'evaluation' in best_result['results'] and best_result['results']['evaluation']:
                eval_info = best_result['results']['evaluation']
                print(f"Test R² Score: {eval_info['test_results']['r2_score']:.4f}")
                print(f"Evaluation plots: {eval_info['model_path'].replace('model.pth', '')}")
    
    print(f"\nGrid Search for {TARGET} regression with {model_type} completed!")
    return all_results

def run_specific_grid_configs(config_indices, model_type, run_evaluation=True, use_target_scaling=True):  # Add this parameter
    """Run specific grid search configurations by their indices (0-based)."""
    
    # Load configurations
    datasets_config = load_config(os.path.join(current_dir, 'datasets_DSST.yaml'))
    splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
    run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))
    
    # Generate all grid search configurations
    configs = generate_grid_search_configs(run_config, splits_config, datasets_config)
    
    results = []
    
    for i in config_indices:
        if i >= len(configs) or i < 0:
            print(f"Warning: Config index {i} is out of range (0-{len(configs)-1}). Skipping.")
            continue
            
        try:
            result = run_single_experiment(configs[i], model_type, run_evaluation, use_target_scaling)  # Add this parameter
            results.append(result)
        except Exception as e:
            print(f"Error in grid search experiment {i + 1}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search regression experiments")
    parser.add_argument("--config", type=int, nargs='+', 
                      help="Specific configuration indices to run (0-based). If not provided, runs all grid combinations.")
    parser.add_argument("--model", type=str, choices=['GraphConv', 'GATv2'],
                      help="Model type to use: GraphConv or GATv2")
    parser.add_argument("--list-configs", action="store_true", 
                      help="List all available grid search configurations and exit")
    parser.add_argument("--no-eval", action="store_true", 
                      help="Skip post-training evaluation")
    parser.add_argument("--no-target-scaling", action="store_true", 
                      help="Disable target scaling")
    
    args = parser.parse_args()
    
    run_evaluation = not args.no_eval
    use_target_scaling = not args.no_target_scaling
    
    if args.list_configs:
        # Load configurations to show available grid search options
        datasets_config = load_config(os.path.join(current_dir, 'datasets_DSST.yaml'))
        splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
        run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))
        
        configs = generate_grid_search_configs(run_config, splits_config, datasets_config)
        
        print("Available Grid Search configurations:")
        print(f"{'Index':<8} {'Dropout':<10} {'WeightDec':<12} {'Layers':<8} {'LR':<10} {'Epochs':<8} {'Train':<8} {'Test':<8} {'Scheduler':<12}")
        print("-" * 105)
        
        for config in configs:
            print(f"{config['config_idx']:<8} {config['dropout_rate']:<10} {config['weight_decay']:<12} {config['layers_num']:<8} {config['lr']:<10} {config['epochs']:<8} {config['split_config']['train']:<8} {config['split_config']['test']:<8} {config['scheduler_config'].get('scheduler', 'none'):<12}")
    
    elif args.config:
        # Run specific configurations
        print(f"Running specific grid search configurations: {args.config} with {args.model} model")
        print(f"Post-training evaluation: {'Enabled' if run_evaluation else 'Disabled'}")
        run_specific_grid_configs(args.config, args.model, run_evaluation, use_target_scaling)  # Add this parameter
    else:
        # Run all grid search configurations
        run_grid_search(args.model, run_evaluation, use_target_scaling)  # Add this parameter