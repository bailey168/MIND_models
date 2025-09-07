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

from gnn.architectures import GraphConvNet
from experiment_regression_DSST import ExperimentRegression

TARGET = 'GF'

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_single_experiment(dataset_config, split_config, run_settings, epochs, scheduler_config, config_idx):
    """Run a single experiment with given configuration."""
    
    print(f"\n{'='*80}")
    print(f"Starting experiment {config_idx + 1}")
    print(f"Learning Rate: {run_settings}")
    print(f"Epochs: {epochs}")
    print(f"Train/Test Split: {split_config['train']}/{split_config['test']}")
    print(f"Scheduler: {scheduler_config.get('scheduler', 'none')}")
    print(f"{'='*80}\n")
    
    # Create model
    model = GraphConvNet(
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
    
    # Prepare optimizer parameters with scheduler
    optim_params = {"lr": run_settings}
    optim_params.update(scheduler_config)
    
    # Create unique experiment ID
    experiment_id = f"{TARGET}_regression_config_{config_idx + 1}_lr_{run_settings}_epochs_{epochs}_scheduler_{scheduler_config.get('scheduler', 'none')}"
    
    # Setup experiment
    experiment = ExperimentRegression(
        sgcn_model=model,  # Using sgcn_model parameter for GCN
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
        id=experiment_id
    )
    
    # Run experiment
    results = experiment.run(num_epochs=epochs, eval_training_set=True)
    
    print(f"\nExperiment {config_idx + 1} completed!")
    print(f"Final Test MSE: SGCN={results['test_sgcn_mse'][-1]:.5f}")
    
    return results

def run_all_dsst_experiments():
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

    # Validate that all lists have the same length
    config_lengths = [len(splits), len(learning_rates), len(epochs_list), len(schedulers)]
    if len(set(config_lengths)) > 1:
        print(f"Warning: Configuration lists have different lengths: {config_lengths}")
        min_length = min(config_lengths)
        print(f"Using first {min_length} configurations from each list")
        splits = splits[:min_length]
        learning_rates = learning_rates[:min_length]
        epochs_list = epochs_list[:min_length]
        schedulers = schedulers[:min_length]
    
    num_experiments = len(splits)
    print(f"Running {num_experiments} experiments total")
    
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
                config_idx=i
            )
            all_results.append({
                'config_idx': i,
                'lr': learning_rates[i],
                'epochs': epochs_list[i],
                'scheduler': schedulers[i].get('scheduler', 'none'),
                'results': results
            })
            
        except Exception as e:
            print(f"Error in experiment {i + 1}: {str(e)}")
            print("Continuing with next experiment...")
            continue
    
    # Print summary of all experiments
    print(f"\n{'='*100}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*100}")
    print(f"{'Config':<8} {'LR':<10} {'Epochs':<8} {'Scheduler':<12} {'Final Test MSE':<15}")
    print("-" * 100)
    
    for result in all_results:
        final_mse = result['results']['test_sgcn_mse'][-1] if result['results']['test_sgcn_mse'] else 'N/A'
        print(f"{result['config_idx'] + 1:<8} {result['lr']:<10} {result['epochs']:<8} {result['scheduler']:<12} {final_mse:<15.5f}")
    
    # Find best performing model
    if all_results:
        valid_results = [r for r in all_results if r['results']['test_sgcn_mse']]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['results']['test_sgcn_mse'][-1])
            print(f"\nBest performing model:")
            print(f"Config {best_result['config_idx'] + 1}: LR={best_result['lr']}, "
                  f"Epochs={best_result['epochs']}, Scheduler={best_result['scheduler']}")
            print(f"Final Test MSE: {best_result['results']['test_sgcn_mse'][-1]:.5f}")
    
    print(f"\n{TARGET} regression experiments completed!")
    return all_results

def run_specific_experiments(config_indices):
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
                config_idx=i
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
    parser.add_argument("--list-configs", action="store_true", 
                      help="List all available configurations and exit")
    
    args = parser.parse_args()
    
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
        print(f"Running specific configurations: {args.config}")
        run_specific_experiments(args.config)
    else:
        # Run all configurations
        run_all_dsst_experiments()