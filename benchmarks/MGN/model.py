from typing import Union

import torch
from dgl import DGLGraph
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNetProcessor, MetaData
from physicsnemo.models.module import Module
from torch import Tensor


class MeshGraphNet(Module):
    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        activation: str = "relu",
    ):
        super().__init__(meta=MetaData())

        if activation == "relu":
            activation_fn = torch.nn.ReLU()
        elif activation == "elu":
            activation_fn = torch.nn.ELU()
        elif activation == "leaky":
            activation_fn = torch.nn.LeakyReLU(0.05)
        else:
            raise ValueError()

        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_edge_encoder,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_node_encoder,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim_node_encoder,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=False,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_node_encoder,
            input_dim_edge=hidden_dim_edge_encoder,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, list[DGLGraph]],
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


def create_model(args):
    model = MeshGraphNet(
        input_dim_nodes=args.input_dim_nodes,
        input_dim_edges=args.input_dim_edges,
        output_dim=args.output_dim,
        processor_size=args.processor_size,
        num_layers_node_processor=args.num_layers_node_processor,
        num_layers_edge_processor=args.num_layers_edge_processor,
        hidden_dim_node_encoder=args.hidden_dim_node_encoder,
        num_layers_node_encoder=args.num_layers_node_encoder,
        hidden_dim_edge_encoder=args.hidden_dim_edge_encoder,
        num_layers_edge_encoder=args.num_layers_edge_encoder,
        hidden_dim_node_decoder=args.hidden_dim_node_decoder,
        num_layers_node_decoder=args.num_layers_node_decoder,
        aggregation=args.aggregation,
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        activation=args.activation,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    return model, optimizer, loss_fn
