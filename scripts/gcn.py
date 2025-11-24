import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.nn import global_mean_pool

import numpy as np

# GCN Model 
class ResGCNBlock(nn.Module):
    """
    Residual GCN block with normalization
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResGCNBlock, self).__init__()
        self.conv     = GCNConv(in_channels, out_channels)
        self.norm     = GraphNorm(out_channels)
        self.dropout  = nn.Dropout(dropout_rate)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, edge_index):
        identity = self.residual(x)
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x + identity


class GCN(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=128, out_channels=256, num_blocks=1): # blocks 5 - 1
        super(GCN, self).__init__()
        
        # Initial transformation
        self.input_transform = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual GCN blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResGCNBlock(hidden_channels, hidden_channels, dropout_rate=0.2))
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.Dropout(0.1)
        )
        
        # L2 normalization
        self.l2_norm = nn.Sequential()  # Using a sequential for consistency

    def siamese_loss(self, Z1, Z2, label, margin=0.3):
        """
        Simplified siamese loss using a combination of cosine distance and contrastive approach. 
        Args:
            Z1, Z2: Pair of feature embeddings
            label : Binary labels (1 for same class, 0 for different class)
            margin: Margin for negative pairs (default: 0.3)    
        Returns:
            Scalar loss value
        """
        # Cosine similarity - values in range [-1, 1] where 1 means identical
        cosine_sim = F.cosine_similarity(Z1, Z2, dim=-1)
        
        # Convert to a distance-like metric: values in range [0, 2] where 0 means identical
        cosine_dist = 1 - cosine_sim
        
        # Simple contrastive loss:
        # - For positive pairs (same class): minimize distance
        # - For negative pairs (different class): push distance beyond margin
        pos_loss = label * cosine_dist
        neg_loss = (1 - label) * F.relu(margin - cosine_dist)
        
        # Return mean loss across the batch
        return torch.mean(pos_loss + neg_loss)
      
    def forward(self, x, edge_index):
        # Ensure inputs are on the same device as the model
        x          = x.to(self.input_transform[0].weight.device)
        edge_index = edge_index.to(self.input_transform[0].weight.device)
        
        # Initial transformation
        x = self.input_transform(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x, edge_index)
        
        # Project and normalize
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)  # L2 normalization
        
        return x
        
   
       