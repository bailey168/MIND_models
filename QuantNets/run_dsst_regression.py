import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to sys.path
def get_root_dir(base: str = "."):
    if any([os.path.isdir(os.path.join(base, child_dir)) and child_dir == "qgrn" for child_dir in os.listdir(base)]):
        return base
    return get_root_dir(base=str(Path(base).parent.resolve()))

root_dir = get_root_dir()
sys.path.append(root_dir)

from gnn.architectures import GCNConvNet
from experiment_regression import ExperimentRegression

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_dsst_experiment():
    # Load configurations
    datasets_config = load_config('datasets_DSST.yaml')
    splits_config = load_config('data.splits_DSST.yaml')
    run_config = load_config('run.settings_DSST.yaml')
    
    dataset_config = datasets_config['dsst_custom']
    
    # Use first split as example
    split_config = splits_config['dsst_custom'][0]
    run_settings = run_config['lrs']['dsst_custom'][0]  # Use first learning rate
    epochs = run_config['epochs']['dsst_custom'][0]     # Use first epoch setting
    
    # Create model
    model = GCNConvNet(
        out_dim=dataset_config['out_dim'],
        input_features=dataset_config['in_channels'],
        output_channels=dataset_config['out_channels'],
        layers_num=dataset_config['layers_num'],
        model_dim=dataset_config['hidden_channels'],
        hidden_sf=dataset_config['gcn_hidden_sf'],
        out_sf=dataset_config['gcn_out_sf'],
        embedding_dim=dataset_config['embedding_dim']
    )
    
    # Setup experiment
    experiment = ExperimentRegression(
        sgcn_model=model,  # Using sgcn_model parameter for GCN
        qgcn_model=None,
        cnn_model=None,
        optim_params={"lr": run_settings},
        base_path=root_dir,
        num_train=split_config['train'],
        num_test=split_config['test'],
        dataset_name=dataset_config['dataset_name'],
        train_batch_size=split_config['batch_size'],
        test_batch_size=split_config['batch_size'],
        train_shuffle_data=True,
        test_shuffle_data=False,
        profile_run=False,
        id=f"dsst_regression_lr_{run_settings}"
    )
    
    # Run experiment
    experiment.run(num_epochs=epochs, eval_training_set=True)
    
    print("DSST regression experiment completed!")

if __name__ == "__main__":
    run_dsst_experiment()