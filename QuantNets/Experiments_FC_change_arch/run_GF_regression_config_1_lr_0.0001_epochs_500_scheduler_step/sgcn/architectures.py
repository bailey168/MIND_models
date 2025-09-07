import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
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
                                    out_channels=hidden_sf * model_dim,
                                    bias=bias,
                                    aggr=aggr
                                    )] + \
                           [GraphConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=hidden_sf * model_dim,
                                    bias=bias,
                                    aggr=aggr
                                    ) for _ in range(layers_num - 2)] + \
                           [GraphConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=out_sf * output_channels,
                                    bias=bias,
                                    aggr=aggr
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Add batch normalization and activation layers
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_sf * model_dim) for _ in range(layers_num - 1)
        ])
        self.activations = torch.nn.ModuleList([
            torch.nn.LeakyReLU() for _ in range(layers_num - 1)
        ])

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels

        if self.include_demo:
            self.demo_projection = torch.nn.Linear(self.demo_dim, graph_features_dim)
            total_features_dim = graph_features_dim * 2
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim * 2),
            torch.nn.BatchNorm1d(model_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim * 2, model_dim),
            torch.nn.BatchNorm1d(model_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
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