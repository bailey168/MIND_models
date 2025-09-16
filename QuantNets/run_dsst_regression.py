import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
from pathlib import Path

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            demo_dim=dataset_config['demo_dim']
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
            demo_dim=dataset_config['demo_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_single_experiment(dataset_config, split_config, run_settings, epochs, scheduler_config, config_idx, model_type, run_evaluation=True, early_stopping_config=None):
    """Run a single experiment with given configuration."""
    
    print(f"\n{'='*80}")
    print(f"Starting experiment {config_idx + 1}")
    print(f"Model Type: {model_type}")
    print(f"Learning Rate: {run_settings}")
    print(f"Epochs: {epochs}")
    print(f"Train/Test Split: {split_config['train']}/{split_config['test']}")
    print(f"Scheduler: {scheduler_config.get('scheduler', 'none')}")
    print(f"Early Stopping: {'Enabled' if early_stopping_config and early_stopping_config.get('enabled', False) else 'Disabled'}")
    print(f"Run Evaluation: {run_evaluation}")
    print(f"{'='*80}\n")
    
    # Create model
    model = create_model(model_type, dataset_config)
    
    # Prepare optimizer parameters with scheduler
    optim_params = {"lr": run_settings}
    optim_params.update(scheduler_config)
    
    # Create unique experiment ID
    early_stop_suffix = "_ES" if early_stopping_config and early_stopping_config.get('enabled', False) else ""
    experiment_id = f"{TARGET}_regression_{model_type}_config_{config_idx + 1}_lr_{run_settings}_epochs_{epochs}_scheduler_{scheduler_config.get('scheduler', 'none')}{early_stop_suffix}"
    
    # Setup experiment
    experiment = ExperimentRegression(
        sgcn_model=model,  # Using sgcn_model parameter for both model types
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
        early_stopping_config=early_stopping_config
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
            'early_stopping_config': early_stopping_config
        }
        config_path = os.path.join(experiment.sgcn_specific_run_dir, "experiment_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f)
        
        # Create a human-readable summary
        summary_path = os.path.join(experiment.sgcn_specific_run_dir, "experiment_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Experiment Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Learning Rate: {run_settings}\n")
            f.write(f"Planned Epochs: {epochs}\n")
            f.write(f"Actual Final Epoch (Saved Model): {results.get('final_sgcn_epoch', epochs)}\n")
            f.write(f"Early Stopped: {'Yes' if results.get('early_stopped', False) else 'No'}\n")
            f.write(f"Scheduler: {scheduler_config.get('scheduler', 'none')}\n")
            f.write(f"Final Test MSE: {results['test_sgcn_mse'][-1]:.5f}\n")
            f.write(f"Best Test MSE: {best_test_mse:.5f} (epoch {best_test_mse_epoch})\n")
            
            if 'evaluation' in results and results['evaluation']:
                eval_results = results['evaluation']
                f.write(f"Test R² Score: {eval_results['test_results']['r2_score']:.4f}\n")
                f.write(f"Test RMSE: {eval_results['test_results']['rmse']:.4f}\n")
                f.write(f"Train R² Score: {eval_results['train_results']['r2_score']:.4f}\n")
    
    print(f"\nExperiment {config_idx + 1} completed!")
    print(f"Planned epochs: {epochs}, Actual final epoch: {results.get('final_sgcn_epoch', epochs)}")
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
    
    return results

def run_all_dsst_experiments(model_type, run_evaluation=True):
    """Run all experiments defined in the YAML configuration files."""
    
    # Load configurations
    datasets_config = load_config(os.path.join(current_dir, 'datasets_DSST.yaml'))
    splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
    run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))

    dataset_config = datasets_config['gf_custom']
    
    # Get all configurations
    splits = splits_config['gf_custom']
    learning_rates = run_config['lrs']['gf_custom']
    epochs_list = run_config['epochs']['gf_custom']
    schedulers = run_config.get('schedulers', {}).get('gf_custom', [{}] * len(learning_rates))
    early_stopping_configs = run_config.get('early_stopping', {}).get('gf_custom', [{}] * len(learning_rates))

    # Validate that all lists have the same length
    config_lengths = [len(splits), len(learning_rates), len(epochs_list), len(schedulers), len(early_stopping_configs)]
    if len(set(config_lengths)) > 1:
        print(f"Warning: Configuration lists have different lengths: {config_lengths}")
        min_length = min(config_lengths)
        print(f"Using first {min_length} configurations from each list")
        splits = splits[:min_length]
        learning_rates = learning_rates[:min_length]
        epochs_list = epochs_list[:min_length]
        schedulers = schedulers[:min_length]
        early_stopping_configs = early_stopping_configs[:min_length]
    
    num_experiments = len(splits)
    print(f"Running {num_experiments} experiments total with {model_type} model")
    print(f"Post-training evaluation: {'Enabled' if run_evaluation else 'Disabled'}")
    
    all_results = []
    
    # Run all experiments
    for i in range(num_experiments):
        try:
            results = run_single_experiment(
                dataset_config=dataset_config,
                split_config=splits[i],
                run_settings=learning_rates[i],
                epochs=epochs_list[i],
                scheduler_config=schedulers[i],
                config_idx=i,
                model_type=model_type,
                run_evaluation=run_evaluation,
                early_stopping_config=early_stopping_configs[i]
            )
            
            # Calculate best test MSE for summary
            best_test_mse = min(results['test_sgcn_mse']) if results['test_sgcn_mse'] else float('inf')
            best_test_mse_epoch = results['test_sgcn_mse'].index(best_test_mse) + 1 if results['test_sgcn_mse'] else 0
            
            all_results.append({
                'config_idx': i,
                'model_type': model_type,
                'lr': learning_rates[i],
                'epochs': epochs_list[i],
                'scheduler': schedulers[i].get('scheduler', 'none'),
                'results': results,
                'best_test_mse': best_test_mse,
                'best_test_mse_epoch': best_test_mse_epoch
            })
            
        except Exception as e:
            print(f"Error in experiment {i + 1}: {str(e)}")
            print("Continuing with next experiment...")
            continue
    
    # Print summary of all experiments
    print(f"\n{'='*120}")
    print(f"SUMMARY OF ALL {model_type} EXPERIMENTS")
    print(f"{'='*120}")
    print(f"{'Config':<8} {'Model':<10} {'LR':<10} {'Epochs':<8} {'Scheduler':<12} {'Final Test MSE':<15} {'Best Test MSE':<15} {'Test R²':<10}")
    print("-" * 120)
    
    for result in all_results:
        final_mse = result['results']['test_sgcn_mse'][-1] if result['results']['test_sgcn_mse'] else 'N/A'
        best_mse = result['best_test_mse'] if result['best_test_mse'] != float('inf') else 'N/A'
        test_r2 = 'N/A'
        if 'evaluation' in result['results'] and result['results']['evaluation']:
            test_r2 = f"{result['results']['evaluation']['test_results']['r2_score']:.4f}"
        
        print(f"{result['config_idx'] + 1:<8} {result['model_type']:<10} {result['lr']:<10} {result['epochs']:<8} {result['scheduler']:<12} {final_mse:<15.5f} {best_mse:<15.5f} {test_r2:<10}")
    
    # Find best performing model
    if all_results:
        valid_results = [r for r in all_results if r['results']['test_sgcn_mse']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['best_test_mse'])
            print(f"\nBest performing model:")
            print(f"Config {best_result['config_idx'] + 1}: Model={best_result['model_type']}, LR={best_result['lr']}, "
                  f"Epochs={best_result['epochs']}, Scheduler={best_result['scheduler']}")
            print(f"Best Test MSE: {best_result['best_test_mse']:.5f} (epoch {best_result['best_test_mse_epoch']})")
            print(f"Final Test MSE: {best_result['results']['test_sgcn_mse'][-1]:.5f}")
            if 'evaluation' in best_result['results'] and best_result['results']['evaluation']:
                eval_info = best_result['results']['evaluation']
                print(f"Test R² Score: {eval_info['test_results']['r2_score']:.4f}")
                print(f"Evaluation plots: {eval_info['model_path'].replace('model.pth', '')}")
    
    print(f"\n{TARGET} regression experiments with {model_type} completed!")
    return all_results

def run_specific_experiments(config_indices, model_type, run_evaluation=True):
    """Run specific experiments by their indices (0-based)."""
    
    # Load configurations
    datasets_config = load_config(os.path.join(current_dir, 'datasets_DSST.yaml'))
    splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
    run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))

    dataset_config = datasets_config['gf_custom']
    
    # Get all configurations
    splits = splits_config['gf_custom']
    learning_rates = run_config['lrs']['gf_custom']
    epochs_list = run_config['epochs']['gf_custom']
    schedulers = run_config.get('schedulers', {}).get('gf_custom', [{}] * len(learning_rates))
    early_stopping_configs = run_config.get('early_stopping', {}).get('gf_custom', [{}] * len(learning_rates))

    results = []
    
    for i in config_indices:
        if i >= len(splits) or i < 0:
            print(f"Warning: Config index {i} is out of range (0-{len(splits)-1}). Skipping.")
            continue
            
        try:
            result = run_single_experiment(
                dataset_config=dataset_config,
                split_config=splits[i],
                run_settings=learning_rates[i],
                epochs=epochs_list[i],
                scheduler_config=schedulers[i],
                config_idx=i,
                model_type=model_type,
                run_evaluation=run_evaluation,
                early_stopping_config=early_stopping_configs[i] if i < len(early_stopping_configs) else {}
            )
            results.append(result)
        except Exception as e:
            print(f"Error in experiment {i + 1}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression experiments")
    parser.add_argument("--config", type=int, nargs='+', 
                      help="Specific configuration indices to run (0-based). If not provided, runs all configs.")
    parser.add_argument("--model", type=str, choices=['GraphConv', 'GATv2'],
                      help="Model type to use: GraphConv or GATv2")
    parser.add_argument("--list-configs", action="store_true", 
                      help="List all available configurations and exit")
    parser.add_argument("--no-eval", action="store_true", 
                      help="Skip post-training evaluation")
    
    args = parser.parse_args()
    
    run_evaluation = not args.no_eval  # Invert the flag
    
    if args.list_configs:
        # Load configurations to show available options
        splits_config = load_config(os.path.join(current_dir, 'data.splits_DSST.yaml'))
        run_config = load_config(os.path.join(current_dir, 'run.settings_DSST.yaml'))
        
        splits = splits_config['gf_custom']
        learning_rates = run_config['lrs']['gf_custom']
        epochs_list = run_config['epochs']['gf_custom']
        schedulers = run_config.get('schedulers', {}).get('gf_custom', [{}] * len(learning_rates))

        print("Available configurations:")
        print(f"{'Index':<8} {'LR':<10} {'Epochs':<8} {'Train':<8} {'Test':<8} {'Scheduler':<12}")
        print("-" * 70)
        
        for i in range(len(splits)):
            scheduler_name = schedulers[i].get('scheduler', 'none') if i < len(schedulers) else 'none'
            print(f"{i:<8} {learning_rates[i]:<10} {epochs_list[i]:<8} {splits[i]['train']:<8} {splits[i]['test']:<8} {scheduler_name:<12}")
    
    elif args.config:
        # Run specific configurations
        print(f"Running specific configurations: {args.config} with {args.model} model")
        print(f"Post-training evaluation: {'Enabled' if run_evaluation else 'Disabled'}")
        run_specific_experiments(args.config, args.model, run_evaluation)
    else:
        # Run all configurations
        run_all_dsst_experiments(args.model, run_evaluation)