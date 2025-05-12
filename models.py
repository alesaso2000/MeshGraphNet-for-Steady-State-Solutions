import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import scatter



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers, do_layer_norm=True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        self.layer_norm = nn.LayerNorm(out_dim)
        self.do_layer_norm = do_layer_norm

    def forward(self, x):
        x = self.net(x)
        if self.do_layer_norm:
            x = self.layer_norm(x)
        return(x)



class Encoder(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, embedding_dim, n_hidden_layers):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, embedding_dim, n_hidden_layers)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, embedding_dim, n_hidden_layers)

    def forward(self, node_features, edge_features):
        h_nodes = self.node_encoder(node_features)
        h_edges = self.edge_encoder(edge_features)
        return h_nodes, h_edges


    
class ProcessorBlock(nn.Module):
    """
    Single message passing block in the processor.
    
    Implements edge and node updates using MLPs and message passing.
    This follows the message passing scheme from the MeshGraphNet paper:
    1. Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
    2. Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e'_ji)
    
    Args:
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, embedding_dim: int, n_mlp_hidden_layers: int):
        super().__init__()
        self.edge_update_mlp = MLP(3 * embedding_dim,  # 3 because node_i, node_j, edge_ij
                                   embedding_dim, 
                                   embedding_dim, 
                                   n_mlp_hidden_layers) 
        self.node_update_mlp = MLP(2 * embedding_dim,   # 2 because node_i, aggregated_edges_{*i}
                                   embedding_dim, 
                                   embedding_dim, 
                                   n_mlp_hidden_layers)  

    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one message passing step.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings [num_nodes, embedding_dim]
            h_edges (torch.Tensor): Edge embeddings [num_edges, embedding_dim]
            edge_index (torch.Tensor): Graph connectivity as [2, num_edges] tensor
                                        where edge_index[0] = sender nodes, edge_index[1] = receiver nodes
            
        Returns:
            tuple: (updated_nodes, updated_edges)
        """
        # Unpack edge indices for sender (i) and receiver (j) nodes
        sender, receiver = edge_index

        # Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
        # Concatenate edge features with sender and receiver node features
        edge_inputs = torch.cat([h_edges, h_nodes[sender], h_nodes[receiver]], dim=-1)
        edge_updates = self.edge_update_mlp(edge_inputs)
        h_edges = h_edges + edge_updates  # Residual connection
        
        # Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e'_ji)
        # First aggregate incoming messages to each node using sum reduction
        # This computes the sum of edge features for each receiving node
        aggregated_edges = scatter(h_edges, receiver, dim=0, dim_size=h_nodes.size(0), reduce='sum')
        # Concatenate node features with aggregated edge features
        node_inputs = torch.cat([h_nodes, aggregated_edges], dim=-1)
        node_updates = self.node_update_mlp(node_inputs)
        h_nodes = h_nodes + node_updates  # Residual connection

        return h_nodes, h_edges



class Processor(nn.Module):
    """
    Multi-layer processor implementing message passing.
    
    Stacks multiple ProcessorBlocks sequentially to perform multiple rounds
    of message passing between nodes.
    
    Args:
        num_layers (int): Number of message passing layers
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, num_layers: int, embedding_dim: int, n_mlp_hidden_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ProcessorBlock(embedding_dim, n_mlp_hidden_layers) for _ in range(num_layers)])

    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process through all layers sequentially.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings
            h_edges (torch.Tensor): Edge embeddings
            edge_index (torch.Tensor): Graph connectivity
            
        Returns:
            tuple: (final_nodes, final_edges)
        """
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, edge_index)
        return h_nodes, h_edges
    




class Decoder(nn.Module):
    """
    Decodes node embeddings to output features.
    
    Uses an MLP without layer normalization for final prediction.
    
    Args:
        embedding_dim (int): Input embedding dimension
        output_dim (int): Output feature dimension
    """
    def __init__(self, embedding_dim: int, output_dim: int, num_hidden_layers: int):
        super().__init__()
        self.mlp = MLP(embedding_dim, embedding_dim, output_dim, num_hidden_layers, do_layer_norm=False)

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        """
        Decode node embeddings to output features.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.mlp(h_nodes)
    


class MeshGraphNet(nn.Module):
    """
    Implementation of MeshGraphNet architecture.
    
    MeshGraphNet is a graph neural network for mesh-based simulations as described in:
    "Learning Mesh-Based Simulation with Graph Networks" by Pfaff et al.
    
    The network has three main components:
    1. Encoder: Maps raw node/edge features to latent embeddings
    2. Processor: Performs multiple rounds of message passing to propagate information
    3. Decoder: Maps final node embeddings to output predictions
    """

    def __init__(self, metadata: dict, sample_graph: Data):
        super().__init__()
        node_dim = sample_graph.x.shape[1]
        edge_dim = sample_graph.edge_attr.shape[1]
        target_dim = sample_graph.target.shape[1]
        self.encoder = Encoder(node_dim, edge_dim, metadata['model']['encoder_mlp_hidden_dim'], 
                               metadata['model']['embedding_dim'], 
                               metadata['model']['n_encoder_mlp_hidden_layers'])
        self.processor = Processor(metadata['model']['n_message_passing_layers'], 
                                   metadata['model']['embedding_dim'],
                                   metadata['model']['n_processor_mlp_hidden_layers'])
        self.decoder = Decoder(metadata['model']['embedding_dim'], 
                               target_dim,  # output is 2 (velocity) + 1 (pressure) dimensional
                               metadata['model']['n_decoder_mlp_hidden_layers'])


    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data (Data): Input graph data containing node fields, type, and edge information
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Process through network
        # 1. Encode: Transform raw features to latent embeddings
        h_nodes, h_edges = self.encoder(graph.x, graph.edge_attr)
        
        # 2. Process: Apply message passing to propagate information
        h_nodes, h_edges = self.processor(h_nodes, h_edges, graph.edge_index)
        
        # 3. Decode: Transform final node embeddings to output predictions
        output = self.decoder(h_nodes)

        return output