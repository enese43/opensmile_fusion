import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class HandcraftedBranch(nn.Module):
    """
    Branch 2: Hand-crafted feature vector processing
    
    Architecture:
    - Fixed weight vector (from .npy) → element-wise multiply (x * fixed)
    - Linear(6373, 256) + LayerNorm + ReLU
    - Output: (N, 256) which gets broadcast to (N, F, 256)
    """
    
    def __init__(self, input_dim: int = 6373, output_dim: int = 128, fixed_weight_npy: Optional[str] = None):
        """
        Initialize the handcrafted features branch.
        
        Args:
            input_dim: Dimension of hand-crafted feature vector
            output_dim: Output feature dimension
            fixed_weight_npy: Path to .npy file with fixed weights (optional)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Fixed weight vector loaded from .npy (if provided)
        if fixed_weight_npy is not None:
            fixed_weights = np.load(fixed_weight_npy)
            assert fixed_weights.shape[0] == input_dim, f"Fixed weights shape {fixed_weights.shape} does not match input_dim {input_dim}"
            self.register_buffer('fixed_weights', torch.from_numpy(fixed_weights).float())
        else:
            self.register_buffer('fixed_weights', torch.ones(input_dim))
        
        # Linear projection layer with LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # Project to output dimensions
            nn.LayerNorm(output_dim),          # Layer normalization
            nn.ReLU()                          # RELU activation
        )
    
    def forward(self, features: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through the handcrafted features branch.
        
        Args:
            features: Hand-crafted feature vector (N, 6373)
            sequence_length: Optional sequence length F for broadcasting
            
        Returns:
            Processed features (N, 128) or (N, F, 128) if sequence_length is provided
        """
        # Apply only fixed weights
        weighted_features = features * self.fixed_weights  # (N, 6373)
        
        # Project through linear layer
        vec_features = self.projection(weighted_features)    # (N, 128)
        
        # Broadcast to sequence length if provided
        if sequence_length is not None:
            vec_features = vec_features.unsqueeze(1).repeat(1, sequence_length, 1)  # (N, F, 128)
        
        return vec_features
    
    def get_feature_dimension(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim 