import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn.architectures import GCNConvNet

# Let's create a simple test to debug the model
def debug_model_and_data():
    print("=== DEBUGGING DSST MODEL ===")
    
    # Load your dataset to inspect
    import pickle
    dataset_path = "/Users/baileyng/MIND_models/QuantNets/Dataset/RawGraph/train_19420_test_4855/custom_dataset_selfloops_True_edgeft_None_norm_True.pkl"
    
    with open(dataset_path, "rb") as f:
        data_struct = pickle.load(f)
    
    # Get a sample graph
    sample_graph = data_struct["geometric"]["sgcn_train_data"][0]
    print(f"Sample graph info:")
    print(f"  Nodes: {sample_graph.num_nodes}")
    print(f"  Edges: {sample_graph.num_edges}")
    print(f"  Node features (x): {sample_graph.x}")
    print(f"  Node features shape: {sample_graph.x.shape}")
    print(f"  Node features dtype: {sample_graph.x.dtype}")
    print(f"  Edge index shape: {sample_graph.edge_index.shape}")
    print(f"  Edge attr shape: {sample_graph.edge_attr.shape}")
    print(f"  Target (y): {sample_graph.y}")
    print(f"  Target dtype: {sample_graph.y.dtype}")
    
    # Check for NaN values in the data
    print(f"\nData quality checks:")
    print(f"  NaN in node features: {torch.isnan(sample_graph.x).any()}")
    print(f"  NaN in edge attributes: {torch.isnan(sample_graph.edge_attr).any()}")
    print(f"  NaN in target: {torch.isnan(sample_graph.y).any()}")
    
    # Check value ranges
    print(f"\nValue ranges:")
    print(f"  Node features min/max: {sample_graph.x.min()}/{sample_graph.x.max()}")
    print(f"  Edge attr min/max: {sample_graph.edge_attr.min()}/{sample_graph.edge_attr.max()}")
    print(f"  Target value: {sample_graph.y.item()}")
    
    # Test the model
    print(f"\n=== TESTING MODEL ===")
    
    # Create model with correct parameters
    model = GCNConvNet(
        out_dim=1,  # Single output for regression
        input_features=21,  # Number of brain regions
        output_channels=64,
        layers_num=3,
        model_dim=64,
        hidden_sf=4,
        out_sf=2,
        embedding_dim=16
    )
    
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample_graph)
            print(f"Model output: {output}")
            print(f"Output shape: {output.shape}")
            print(f"Output contains NaN: {torch.isnan(output).any()}")
            
            # Test loss computation
            loss = F.mse_loss(output, sample_graph.y.float())
            print(f"Loss: {loss.item()}")
            print(f"Loss is NaN: {torch.isnan(loss)}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_and_data()