import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
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
                embedding_dim=16, include_demo=True, demo_dim=4, dropout_rate=0.6):
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

        # Use consistent dimensions: 64 for all layers (matching GATv2ConvNet)
        hidden_dim = 64

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
            # Sequential MLPs for demographic data - first layer takes raw demo data, subsequent layers take previous MLP output
            self.demo_mlps = torch.nn.ModuleList()
            for i in range(self.layers_num):
                if i == 0:
                    # First layer takes raw demographic data
                    self.demo_mlps.append(torch.nn.Linear(self.demo_dim, 16))
                else:
                    # Subsequent layers take output from previous demo MLP (16 dimensions)
                    self.demo_mlps.append(torch.nn.Linear(16, 16))
            
            # 2-layer MLP to downsize concatenated features
            self.downsize_mlps = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 16, hidden_dim * 2),
                    torch.nn.LeakyReLU(),
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

        # Add AttentionalAggregation instead of global_mean_pool
        attention_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        self.global_attention_pool = AttentionalAggregation(gate_nn=attention_gate)

        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.x = self.input_projection(data.x)

        # Initialize demo_features for sequential processing
        demo_features = None
        if self.include_demo and hasattr(data, 'demographics'):
            demo_features = data.demographics

        for i in range(self.layers_num):
            # Apply graph convolution
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)
            
            # Only concatenate demographics for layers except the last one
            if self.include_demo and demo_features is not None and i < self.layers_num - 1:
                # Process demographics through sequential MLPs
                demo_features = self.demo_mlps[i](demo_features)
                
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

        # Use attentional aggregation instead of global mean pooling
        graph_features = self.global_attention_pool(data.x, data.batch)
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
        out_channels_per_head = 16   # when concat=True
        heads = 4
        hidden_dim = out_channels_per_head * heads

        self.input_projection = torch.nn.Linear(embedding_dim, hidden_dim)

        # All conv layers have same input/output dimensions
        self.conv_layers = torch.nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                bias=bias,
                edge_dim=1,
                residual=True,
                dropout=self.dropout_rate,
                concat=False
            ) for _ in range(layers_num)
        ])

        if self.include_demo:
            # Sequential MLPs for demographic data - first layer takes raw demo data, subsequent layers take previous MLP output
            self.demo_mlps = torch.nn.ModuleList()
            for i in range(self.layers_num):
                if i == 0:
                    # First layer takes raw demographic data
                    self.demo_mlps.append(torch.nn.Linear(self.demo_dim, 16))
                else:
                    # Subsequent layers take output from previous demo MLP (16 dimensions)
                    self.demo_mlps.append(torch.nn.Linear(16, 16))

            # 2-layer MLP to downsize concatenated features
            self.downsize_mlps = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 16, hidden_dim * 2),
                    torch.nn.LeakyReLU(),
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

        # Add AttentionalAggregation instead of global_mean_pool
        attention_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        self.global_attention_pool = AttentionalAggregation(gate_nn=attention_gate)

        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.x = self.input_projection(data.x)

        # Initialize demo_features for sequential processing
        demo_features = None
        if self.include_demo and hasattr(data, 'demographics'):
            demo_features = data.demographics

        for i in range(self.layers_num):
            # Apply graph convolution
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
            
            # Only concatenate demographics for layers except the last one
            if self.include_demo and demo_features is not None and i < self.layers_num - 1:
                # Process demographics through sequential MLPs
                demo_features = self.demo_mlps[i](demo_features)
                
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

        # Use attentional aggregation instead of global mean pooling
        graph_features = self.global_attention_pool(data.x, data.batch)
        x = self.classifier(graph_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)