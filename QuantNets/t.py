import torch
data = torch.load('/Users/baileyng/MIND_data/DSST_FC_graph/1000184_25752_2_0_fc_graph.pt')
print(f"Node features shape: {data.x.shape}")  # Should be [n_nodes, n_features]
print(f"Edge weights shape: {data.edge_attr.shape}")  # Should be [n_edges, 1]