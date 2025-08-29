import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, SGConv, GENConv, GeneralConv, GATv2Conv, TransformerConv

# GCNConv. 2016 https://arxiv.org/abs/1609.02907
# class GCNConv(
#   in_channels: int, 
#   out_channels: int, 
#   improved: bool = False, 
#   cached: bool = False, 
#   add_self_loops: Optional[bool] = None, 
#   normalize: bool = True, 
#   bias: bool = True, **kwargs)
class GCNConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=5, out_sf=4, improved=True, cached=False, aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=5):
        super(GCNConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        # Add embedding layer to convert node indices to dense features
        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [GCNConv(
                                    in_channels=embedding_dim,
                                    out_channels=hidden_sf * model_dim,
                                    improved=improved,
                                    cached=cached,
                                    aggr=aggr
                                    )] + \
                           [GCNConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=hidden_sf * model_dim,
                                    improved=improved,
                                    cached=cached,
                                    aggr=aggr
                                    ) for _ in range(layers_num - 2)] + \
                           [GCNConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=out_sf * output_channels,
                                    improved=improved,
                                    cached=cached,
                                    aggr=aggr
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
    

# ChebConv. 2016 https://arxiv.org/abs/1606.09375
# class GCNConv(
#   in_channels: int, 
#   out_channels: int, 
#   K: int, 
#   normalization: Optional[str] = 'sym', 
#   bias: bool = True, **kwargs)
class ChebConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=3, out_sf=2, K=3, normalization='sym', aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=5): # norm: sym / rw / None
        super(ChebConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [ChebConv(
                                    in_channels=embedding_dim,
                                    out_channels=hidden_sf * model_dim,
                                    K=K,
                                    normalization=normalization,
                                    aggr=aggr
                                    )] + \
                           [ChebConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=hidden_sf * model_dim,
                                    K=K,
                                    normalization=normalization,
                                    aggr=aggr
                                    ) for _ in range(layers_num - 2)] + \
                           [ChebConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=out_sf * output_channels,
                                    K=K,
                                    normalization=normalization,
                                    aggr=aggr
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight, batch=data.batch)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
    

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
                embedding_dim=16, include_demo=True, demo_dim=5):
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

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)


# SGConv. 2019 https://arxiv.org/abs/1902.07153
# class SGConv(
#   in_channels: int, 
#   out_channels: int, 
#   K: int = 1, 
#   cached: bool = False, 
#   add_self_loops: bool = True, 
#   bias: bool = True, 
#   **kwargs)
class SGConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=5, out_sf=4, K=5, bias=True, aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=5):
        super(SGConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [SGConv(
                                    in_channels=embedding_dim,
                                    out_channels=hidden_sf * model_dim,
                                    K=K,
                                    bias=bias,
                                    aggr=aggr
                                    )] + \
                           [SGConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=hidden_sf * model_dim,
                                    K=K,
                                    bias=bias,
                                    aggr=aggr
                                    ) for _ in range(layers_num - 2)] + \
                           [SGConv(
                                    in_channels=hidden_sf * model_dim,
                                    out_channels=out_sf * output_channels,
                                    K=K,
                                    bias=bias,
                                    aggr=aggr
                                    )]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_weight = data.edge_attr.squeeze(-1)
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_weight=edge_weight)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
 

# GENConv 2020 https://arxiv.org/abs/2006.07739
# class GENConv(in_channels: Union[int, Tuple[int, int]], 
#               out_channels: int, 
#               aggr: Optional[Union[str, List[str], Aggregation]] = 'softmax', 
#               t: float = 1.0, 
#               learn_t: bool = False, 
#               p: float = 1.0, 
#               learn_p: bool = False, 
#               msg_norm: bool = False, 
#               learn_msg_scale: bool = False, 
#               norm: str = 'batch', 
#               num_layers: int = 2, 
#               expansion: int = 2, 
#               eps: float = 1e-07, 
#               bias: bool = False, 
#               edge_dim: Optional[int] = None, **kwargs)
class GENConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=3, out_sf=2, use_bias=True, aggr='add',
                embedding_dim=16, include_demo=True, demo_dim=5):
        super(GENConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [GENConv(
                                      in_channels=embedding_dim,
                                      out_channels=hidden_sf * model_dim,
                                      num_layers=1,
                                      bias=use_bias,
                                      aggr=aggr
                                      )] + \
                           [GENConv(
                                      in_channels=hidden_sf * model_dim,
                                      out_channels=hidden_sf * model_dim,
                                      num_layers=1,
                                      bias=use_bias,
                                      aggr=aggr
                                      ) for _ in range(layers_num - 2)] + \
                           [GENConv(
                                      in_channels=hidden_sf * model_dim,
                                      out_channels=out_sf * output_channels,
                                      num_layers=1,
                                      bias=use_bias,
                                      aggr=aggr
                                      )]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)


# GeneralConv 2020 https://arxiv.org/abs/2011.08843
# class GeneralConv(in_channels: Union[int, Tuple[int, int]], 
#                   out_channels: Optional[int], 
#                   in_edge_channels: Optional[int] = None, 
#                   aggr: str = 'add', 
#                   skip_linear: str = False, 
#                   directed_msg: bool = True, 
#                   heads: int = 1, 
#                   attention: bool = False, 
#                   attention_type: str = 'additive', 
#                   l2_normalize: bool = False, 
#                   bias: bool = True, **kwargs)
class GeneralConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=3, out_sf=1, hidden_heads=4, use_bias=True, aggr="add",
                embedding_dim=16, include_demo=True, demo_dim=5):
        super(GeneralConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [GeneralConv(
                                      in_channels=embedding_dim,
                                      out_channels=hidden_sf * model_dim,
                                      heads=hidden_heads,
                                      bias=use_bias,
                                      aggr=aggr
                                      )] + \
                           [GeneralConv(
                                      in_channels=hidden_sf * model_dim,
                                      out_channels=hidden_sf * model_dim,
                                      heads=hidden_heads,
                                      bias=use_bias,
                                      aggr=aggr
                                      ) for _ in range(layers_num - 2)] + \
                           [GeneralConv(
                                      in_channels=hidden_sf * model_dim,
                                      out_channels=out_sf * output_channels,
                                      heads=hidden_heads,
                                      bias=use_bias,
                                      aggr=aggr
                                      )]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
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
                model_dim, hidden_sf=1, out_sf=1, hidden_heads=3, use_bias=True, aggr="add",
                embedding_dim=16, include_demo=True, demo_dim=5):
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
                                      out_channels=hidden_sf * model_dim,
                                      heads=hidden_heads,
                                      bias=use_bias,
                                      aggr=aggr
                                      )] + \
                           [GATv2Conv(
                                      in_channels=hidden_sf * hidden_heads * model_dim,
                                      out_channels=hidden_sf * hidden_heads * model_dim,
                                      heads=(hidden_heads - 1),
                                      bias=use_bias,
                                      aggr=aggr
                                      ) for _ in range(layers_num - 2)] + \
                           [GATv2Conv(
                                      in_channels=hidden_sf * (hidden_heads - 1) * hidden_heads * model_dim,
                                      out_channels=out_sf * output_channels,
                                      heads=1,
                                      bias=use_bias,
                                      aggr=aggr
                                      )]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)

        for i in range(self.layers_num):
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
   

# TransformerConv 2020 https://arxiv.org/abs/2009.03509
# class TransformerConv(
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout: float = 0.,
#         edge_dim: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     )
class TransformerConvNet(torch.nn.Module):
    def __init__(self, out_dim, input_features, output_channels, layers_num, 
                model_dim, hidden_sf=1, out_sf=1, hidden_heads=3, use_bias=True, aggr="add",
                embedding_dim=16, include_demo=True, demo_dim=5):
        super(TransformerConvNet, self).__init__()
        self.layers_num = layers_num
        self.out_dim = out_dim  # Store output dimension
        self.include_demo = include_demo
        self.demo_dim = demo_dim

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=input_features,
            embedding_dim=embedding_dim
        )

        self.conv_layers = [TransformerConv(
                                      in_channels=embedding_dim,
                                      out_channels=hidden_sf * model_dim,
                                      heads=hidden_heads,
                                      bias=use_bias,
                                      aggr=aggr
                                      )] + \
                           [TransformerConv(
                                      in_channels=hidden_sf * hidden_heads * model_dim,
                                      out_channels=hidden_sf * hidden_heads * model_dim,
                                      heads=max(1, hidden_heads - 2),
                                      bias=use_bias,
                                      aggr=aggr
                                      ) for _ in range(layers_num - 2)] + \
                           [TransformerConv(
                                      in_channels=hidden_sf * max(1, hidden_heads - 2) * hidden_heads * model_dim,
                                      out_channels=out_sf * output_channels,
                                      heads=1,
                                      bias=use_bias,
                                      aggr=aggr
                                      )]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        # Calculate final feature dimension
        graph_features_dim = out_sf * output_channels
        if self.include_demo:
            total_features_dim = graph_features_dim + self.demo_dim
        else:
            total_features_dim = graph_features_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(total_features_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model_dim, out_dim)
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        
        for i in range(self.layers_num):
            edge_attr = data.edge_attr
            data.x = self.conv_layers[i](data.x, data.edge_index, edge_attr=edge_attr)
        graph_features = global_mean_pool(data.x, data.batch)

        # Concatenate with demographic features if available
        if self.include_demo and hasattr(data, 'demographics'):
            combined_features = torch.cat([graph_features, data.demographics], dim=1)
        else:
            combined_features = graph_features

        x = self.classifier(combined_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)
