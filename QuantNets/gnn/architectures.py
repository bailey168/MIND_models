import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv, GATv2Conv

# GraphConv. 2018 https://arxiv.org/abs/1810.02244
# class GCNConv(
#   in_channels: Union[int, Tuple[int, int]], 
#   out_channels: int, 
#   aggr: str = 'add', 
#   bias: bool = True, 
#   **kwargs)
class GraphConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=4, out_sf=2, bias=True, aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=4, dropout_rate=0.1):
        super(GraphConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim
        self.dropout_rate = dropout_rate

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        hidden_dim = 2 * model_dim

        # Project embedding to hidden dimension
        self.input_projection = torch.nn.Linear(embedding_dim, hidden_dim)

        self.conv_layers = torch.nn.ModuleList([
            GraphConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                bias=bias,
                aggr=aggr
            ) for _ in range(self.layers_num)
        ])

        if self.include_demo:
            # 1-layer MLP for demographic data at each layer
            self.demo_mlps = torch.nn.ModuleList([
                torch.nn.Linear(self.demo_dim, 16) for _ in range(self.layers_num)
            ])
            
            # 2-layer MLP to downsize concatenated features
            self.downsize_mlps = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 16, hidden_dim * 2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(self.dropout_rate),
                    torch.nn.Linear(hidden_dim * 2, hidden_dim)
                ) for _ in range(self.layers_num)
            ])

        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.GraphNorm(hidden_dim) for _ in range(layers_num - 1)
        ])
        self.activations = torch.nn.ModuleList([
            torch.nn.LeakyReLU() for _ in range(layers_num - 1)
        ])

        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.x = self.input_projection(data.x)

        for i in range(self.layers_num):
            # Apply graph convolution
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)
            
            if self.include_demo and hasattr(data, 'demographics'):
                # Process demographics through 1-layer MLP
                demo_features = self.demo_mlps[i](data.demographics)
                
                # Expand demographics to match number of nodes
                nodes_per_graph = torch.bincount(data.batch)
                demo_expanded = torch.repeat_interleave(demo_features, nodes_per_graph, dim=0)

                # Concatenate graph features with demographic features
                combined = torch.cat([data.x, demo_expanded], dim=1)

                # Downsize through 2-layer MLP
                data.x = self.downsize_mlps[i](combined)
            
            # Apply normalization and activation (except for last layer)
            if i < self.layers_num - 1:
                data.x = self.batch_norms[i](data.x)
                data.x = self.activations[i](data.x)


        # Global pooling and regression
        graph_features = global_mean_pool(data.x, data.batch)
        x = self.classifier(graph_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
        

# GATv2Conv 2021 https://arxiv.org/abs/2105.14491
# class GATv2Conv(in_channels: Union[int, Tuple[int, int]], 
#                 out_channels: int, 
#                 heads: int = 1, 
#                 concat: bool = True, 
#                 negative_slope: float = 0.2, 
#                 dropout: float = 0.0, 
#                 add_self_loops: bool = True, 
#                 edge_dim: Optional[int] = None, 
#                 fill_value: Union[float, Tensor, str] = 'mean', 
#                 bias: bool = True, 
#                 share_weights: bool = False)
class GATv2ConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=4, out_sf=2, hidden_heads=4, bias=True, aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=4, dropout_rate=0.6):
        super(GATv2ConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim
        self.dropout_rate = dropout_rate

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        # Use consistent dimensions: 16 * 4 = 64 for all layers
        out_channels_per_head = 16
        heads = 4
        hidden_dim = out_channels_per_head * heads

        if self.include_demo:
            # 1-layer MLP for demographic data (initial)
            self.initial_demo_mlp = torch.nn.Linear(self.demo_dim, 16)
            
            # 2-layer MLP to process concatenated embeddings + demographics before first graph layer
            self.initial_downsize_mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim + 16, hidden_dim * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            self.input_projection = torch.nn.Linear(embedding_dim, hidden_dim)

        # All conv layers have same input/output dimensions
        self.conv_layers = torch.nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=out_channels_per_head,
                heads=heads,
                bias=bias,
                edge_dim=1,
                residual=True,
                dropout=self.dropout_rate
            ) for _ in range(layers_num)
        ])

        if self.include_demo:
            # 1-layer MLP for demographic data at each layer
            self.demo_mlps = torch.nn.ModuleList([
                torch.nn.Linear(self.demo_dim, 16) for _ in range(self.layers_num)
            ])

            # 2-layer MLP to downsize concatenated features
            self.downsize_mlps = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 16, hidden_dim * 2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(self.dropout_rate),
                    torch.nn.Linear(hidden_dim * 2, hidden_dim)
                ) for _ in range(self.layers_num)
            ])


        # Add batch normalization layers
        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.GraphNorm(hidden_dim) for _ in range(layers_num - 1)
        ])

        self.activations = torch.nn.ModuleList([
            torch.nn.ELU() for _ in range(layers_num - 1)
        ])

        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        
        # Process initial embeddings + demographics before first graph layer
        if self.include_demo and hasattr(data, 'demographics'):
            # Process demographics through initial 1-layer MLP
            demo_features = self.initial_demo_mlp(data.demographics)
            
            # Expand demographics to match number of nodes
            nodes_per_graph = torch.bincount(data.batch)
            demo_expanded = torch.repeat_interleave(demo_features, nodes_per_graph, dim=0)
            
            # Concatenate node embeddings with demographic features
            combined_initial = torch.cat([data.x, demo_expanded], dim=1)
            
            # Process through 2-layer MLP
            data.x = self.initial_downsize_mlp(combined_initial)
        else:
            data.x = self.input_projection(data.x)

        for i in range(self.layers_num):
            # Apply graph convolution
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
            
            if self.include_demo and hasattr(data, 'demographics'):
                # Process demographics through 1-layer MLP
                demo_features = self.demo_mlps[i](data.demographics)
                
                # Expand demographics to match number of nodes
                nodes_per_graph = torch.bincount(data.batch)
                demo_expanded = torch.repeat_interleave(demo_features, nodes_per_graph, dim=0)
                
                # Concatenate graph features with demographic features
                combined = torch.cat([data.x, demo_expanded], dim=1)
                
                # Downsize through 2-layer MLP
                data.x = self.downsize_mlps[i](combined)

            # Apply normalization (except for last layer)
            if i < self.layers_num - 1:
                data.x = self.batch_norms[i](data.x)
                data.x = self.activations[i](data.x)

        # Global pooling and regression
        graph_features = global_mean_pool(data.x, data.batch)
        x = self.classifier(graph_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)