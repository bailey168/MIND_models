import torch
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from gnn.architectures import GraphConvNet, GATv2ConvNet

def count_parameters(model, print_details=False):
    """
    Count the total number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        print_details: If True, prints parameter count for each layer
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0
    
    if print_details:
        print(f"\n{'Layer Name':<40} {'Parameters':<15} {'Trainable':<10}")
        print("-" * 65)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            trainable_str = "Yes"
        else:
            trainable_str = "No"
        
        if print_details:
            print(f"{name:<40} {param_count:<15,} {trainable_str:<10}")
    
    if print_details:
        print("-" * 65)
        print(f"{'Total Parameters:':<40} {total_params:<15,}")
        print(f"{'Trainable Parameters:':<40} {trainable_params:<15,}")
        print(f"{'Non-trainable Parameters:':<40} {total_params - trainable_params:<15,}")
    
    return total_params, trainable_params

def create_model_from_config(model_type, config):
    """Create model instance from configuration."""
    if model_type == 'GraphConv':
        return GraphConvNet(
            out_dim=config['out_dim'],
            input_features=config['in_channels'],
            output_channels=config['out_channels'],
            layers_num=config['layers_num'],
            model_dim=config['hidden_channels'],
            hidden_sf=config['graph_hidden_sf'],
            out_sf=config['graph_out_sf'],
            embedding_dim=config['embedding_dim'],
            include_demo=config['include_demo'],
            demo_dim=config['demo_dim'],
            dropout_rate=config.get('dropout_rate', 0.6)
        )
    elif model_type == 'GATv2':
        return GATv2ConvNet(
            out_dim=config['out_dim'],
            input_features=config['in_channels'],
            output_channels=config['out_channels'],
            layers_num=config['layers_num'],
            model_dim=config['hidden_channels'],
            hidden_sf=config['gatv2_hidden_sf'],
            out_sf=config['gatv2_out_sf'],
            hidden_heads=config['gatv2_hidden_heads'],
            embedding_dim=config['embedding_dim'],
            include_demo=config['include_demo'],
            demo_dim=config['demo_dim'],
            dropout_rate=config.get('dropout_rate', 0.6)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024 / 1024
    return model_size

def compare_model_configurations():
    """Compare parameter counts across different model configurations."""
    
    # Base configuration from your datasets_DSST.yaml
    base_config = {
        'out_dim': 1,
        'in_channels': 21,
        'out_channels': 32,
        'hidden_channels': 32,
        'embedding_dim': 16,
        'include_demo': True,
        'demo_dim': 4,
        'graph_hidden_sf': 1,
        'graph_out_sf': 2,
        'gatv2_hidden_sf': 2,
        'gatv2_out_sf': 1,
        'gatv2_hidden_heads': 4,
        'dropout_rate': 0.6
    }
    
    # Test different layer configurations
    layer_configs = [2, 3, 4]
    model_types = ['GraphConv', 'GATv2']
    
    print("="*100)
    print("MODEL PARAMETER COMPARISON")
    print("="*100)
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{model_type} Model:")
        print("-" * 50)
        print(f"{'Layers':<8} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<12}")
        print("-" * 50)
        
        results[model_type] = {}
        
        for layers in layer_configs:
            config = base_config.copy()
            config['layers_num'] = layers
            
            model = create_model_from_config(model_type, config)
            total_params, trainable_params = count_parameters(model)
            model_size = get_model_size_mb(model)
            
            results[model_type][layers] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'size_mb': model_size
            }
            
            print(f"{layers:<8} {total_params:<15,} {trainable_params:<15,} {model_size:<12.2f}")
    
    return results

def analyze_specific_model(model_type, layers_num, print_details=True):
    """Analyze a specific model configuration in detail."""
    
    # Configuration from your datasets_DSST.yaml
    config = {
        'out_dim': 1,
        'in_channels': 21,
        'out_channels': 32,
        'layers_num': layers_num,
        'hidden_channels': 32,
        'embedding_dim': 16,
        'include_demo': True,
        'demo_dim': 4,
        'graph_hidden_sf': 1,
        'graph_out_sf': 2,
        'gatv2_hidden_sf': 2,
        'gatv2_out_sf': 1,
        'gatv2_hidden_heads': 4,
        'dropout_rate': 0.6
    }
    
    print("="*100)
    print(f"DETAILED ANALYSIS: {model_type} Model with {layers_num} layers")
    print("="*100)
    
    model = create_model_from_config(model_type, config)
    total_params, trainable_params = count_parameters(model, print_details=print_details)
    model_size = get_model_size_mb(model)
    
    print(f"\nModel Summary:")
    print(f"- Architecture: {model_type}")
    print(f"- Number of layers: {layers_num}")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Model size: {model_size:.2f} MB")
    print(f"- Include demographics: {config['include_demo']}")
    print(f"- Demographic dimensions: {config['demo_dim']}")
    
    return model, total_params, trainable_params, model_size

def main():
    """Main function to run parameter counting analysis."""
    
    print("Graph Neural Network Model Parameter Analysis")
    print("="*100)
    
    # 1. Compare different configurations
    comparison_results = compare_model_configurations()
    
    # 2. Detailed analysis of specific models (matching your grid search)
    print("\n" + "="*100)
    print("DETAILED ANALYSIS FOR GRID SEARCH CONFIGURATIONS")
    print("="*100)
    
    # Analyze the configurations from your grid search
    grid_layers = [2, 3, 4]  # From your run.settings_DSST.yaml
    
    for model_type in ['GraphConv', 'GATv2']:
        for layers in grid_layers:
            print(f"\n{'-'*60}")
            print(f"Configuration: {model_type} with {layers} layers")
            print(f"{'-'*60}")
            
            model, total, trainable, size = analyze_specific_model(
                model_type, layers, print_details=False
            )
            
            # Brief summary
            print(f"Parameters: {total:,} | Size: {size:.2f} MB")
    
    # 3. Show the impact of demographics
    print(f"\n{'='*100}")
    print("IMPACT OF DEMOGRAPHIC FEATURES")
    print("="*100)
    
    for model_type in ['GraphConv', 'GATv2']:
        print(f"\n{model_type} Model (3 layers):")
        
        # With demographics
        config_with_demo = {
            'out_dim': 1, 'in_channels': 21, 'out_channels': 32, 'layers_num': 3,
            'hidden_channels': 32, 'embedding_dim': 16, 'include_demo': True,
            'demo_dim': 4, 'graph_hidden_sf': 1, 'graph_out_sf': 2,
            'gatv2_hidden_sf': 2, 'gatv2_out_sf': 1, 'gatv2_hidden_heads': 4,
            'dropout_rate': 0.6
        }
        
        # Without demographics
        config_without_demo = config_with_demo.copy()
        config_without_demo['include_demo'] = False
        
        model_with = create_model_from_config(model_type, config_with_demo)
        model_without = create_model_from_config(model_type, config_without_demo)
        
        total_with, _ = count_parameters(model_with)
        total_without, _ = count_parameters(model_without)
        
        demo_params = total_with - total_without
        
        print(f"  With demographics:    {total_with:,} parameters")
        print(f"  Without demographics: {total_without:,} parameters")
        print(f"  Demographics add:     {demo_params:,} parameters ({demo_params/total_with*100:.1f}%)")

if __name__ == "__main__":
    main()