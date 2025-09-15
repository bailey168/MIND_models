import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, SGConv, GENConv, GeneralConv, GATv2Conv, TransformerConv

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
                embedding_dim=16, include_demo=True, demo_dim=4):
        super(GraphConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [GraphConv(
                                    in_channels=embedding_dim,
                                    out_channels=1 * model_dim,
                                    bias=bias,
                                    aggr=aggr
                                    )] + \
                           [GraphConv(
                                    in_channels=1 * model_dim,
                                    out_channels=2 * model_dim,
                                    bias=bias,
                                    aggr=aggr
                                    )] + \
                           [GraphConv(
                                    in_channels=2 * model_dim,
                                    out_channels=4 * model_dim,
                                    bias=bias,
                                    aggr=aggr
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Add batch normalization and activation layers
        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.GraphNorm(1 * model_dim),
            pyg_nn.norm.GraphNorm(2 * model_dim)
        ])
        self.activations = torch.nn.ModuleList([
            torch.nn.LeakyReLU(),
            torch.nn.LeakyReLU()
        ])

        # Calculate final feature dimension
        graph_features_dim = 4 * model_dim

        if self.include_demo:
            self.demo_projection = torch.nn.Linear(self.demo_dim, 16)
            total_features_dim = graph_features_dim + 16
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim * 2),
            torch.nn.BatchNorm1d(model_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(model_dim * 2, model_dim),
            torch.nn.BatchNorm1d(model_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)

            if i < self.layers_num - 1:
                data.x = self.batch_norms[i](data.x)
                data.x = self.activations[i](data.x)

        graph_features = global_mean_pool(data.x, data.batch)

        # Process demographic features through linear layer and concatenate
        if self.include_demo and hasattr(data, 'demographics'):
            demo_features = self.demo_projection(data.demographics)
            combined_features = torch.cat([graph_features, demo_features], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

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
                embedding_dim=16, include_demo=True, demo_dim=4):
        super(GATv2ConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [GATv2Conv(
                                    in_channels=embedding_dim,
                                    out_channels=16,
                                    heads=4,
                                    bias=bias,
                                    edge_dim=1,
                                    residual=True,
                                    dropout=0.6
                                    )] + \
                           [GATv2Conv(
                                    in_channels=64,
                                    out_channels=16,
                                    heads=4,
                                    bias=bias,
                                    edge_dim=1,
                                    residual=True,
                                    dropout=0.6
                                    )] + \
                           [GATv2Conv(
                                    in_channels=64,
                                    out_channels=16,
                                    heads=4,
                                    bias=bias,
                                    edge_dim=1,
                                    residual=True,
                                    dropout=0.6
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Add batch normalization and activation layers
        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.GraphNorm(64) for _ in range(layers_num - 1)
        ])
        self.activations = torch.nn.ModuleList([
            torch.nn.ELU() for _ in range(layers_num - 1)
        ])

        # Calculate final feature dimension
        graph_features_dim = 64

        if self.include_demo:
            self.demo_projection = torch.nn.Linear(self.demo_dim, 16)
            total_features_dim = graph_features_dim + 16
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(32, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)

            if i < self.layers_num - 1:
                data.x = self.batch_norms[i](data.x)
                data.x = self.activations[i](data.x)

        graph_features = global_mean_pool(data.x, data.batch)

        # Process demographic features through linear layer and concatenate
        if self.include_demo and hasattr(data, 'demographics'):
            demo_features = self.demo_projection(data.demographics)
            combined_features = torch.cat([graph_features, demo_features], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)