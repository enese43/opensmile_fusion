import torch
import torch.nn as nn
from typing import List, Dict, Any


class GatedFusion(nn.Module):
    """
    Gated Fusion for multimodal audio spoofing detection
    
    Architecture:
    1. Generate gating values (alpha, beta) from XLS-R features
    2. Scale each branch's features with their respective gates
    3. Sum the scaled features for final fusion
    4. Maintain sequence dimension: (N, F, 256)
    """
    
    def __init__(self, 
                 branch_dims: List[int],
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 gate_type: str = "scalar"):
        """
        Initialize the gated fusion module.
        
        Args:
            branch_dims: List of feature dimensions from each branch [cnn_dim, handcrafted_dim, xlsr_dim]
            hidden_dim: Hidden dimension in the gate network
            num_classes: Number of output classes (not used in this version)
            dropout_rate: Dropout rate in the gate network and fusion
            gate_type: Type of gates ("scalar" or "vector")
        """
        super().__init__()
        
        self.branch_dims = branch_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gate_type = gate_type
        self.dropout_rate = dropout_rate
        
        # Assuming all branches output the same dimension for simplicity
        self.common_dim = branch_dims[0]  # Use first branch dimension as common (256)
        
        # Gate generation network (from XLS-R features)
        # Generate 2 gates: alpha and beta
        self.gate_network = nn.Sequential(
            nn.Linear(branch_dims[2], hidden_dim),  # XLS-R dim (256) → hidden
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Generate 2 gates (alpha, beta)
        )
        
        # Optional projection layers if branch dimensions differ
        self.projections = nn.ModuleList()
        for i, dim in enumerate(branch_dims):
            if dim != self.common_dim:
                self.projections.append(nn.Linear(dim, self.common_dim))
            else:
                self.projections.append(nn.Identity())
        
        # Add dropout after fusion
        self.fusion_dropout = nn.Dropout(dropout_rate)
        
        # Add normalization after fusion
        self.fusion_norm = nn.LayerNorm(self.common_dim)
        
        # Store fusion type for reference
        self.fusion_type = "gated"
    
    def forward(self, branch_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the gated fusion.
        
        Args:
            branch_features: List of feature tensors from each branch
                           [cnn_features, handcrafted_features, xlsr_features]
                           All should be (N, F, 256) where F is sequence length
            
        Returns:
            Fused features (N, F, 256)
        """
        cnn_features, handcrafted_features, xlsr_features = branch_features
        
        # All inputs should be (N, F, 256)
        N, F, D = xlsr_features.shape
        
        # Generate gating values from XLS-R features
        # Use mean pooling of XLS-R features to generate gates
        xlsr_pooled = torch.mean(xlsr_features, dim=1)  # (N, 256)
        gates = self.gate_network(xlsr_pooled)  # (N, 2)
        # alpha, beta = torch.split(gates, 1, dim=1)  # Each: (N, 1)
        
        gates_combined = torch.cat([alpha, beta], dim=1)  # (N, 2)
        gates_softmax = torch.softmax(gates_combined, dim=1)  # (N, 2)
        alpha, beta = torch.split(gates_softmax, 1, dim=1)  # Each: (N, 1)
        
        cnn_projected = self.projections[0](cnn_features)      # (N, F, 256)
        handcrafted_projected = self.projections[1](handcrafted_features)  # (N, F, 256)
        xlsr_projected = self.projections[2](xlsr_features)    # (N, F, 256)
        
        # Apply gates (broadcast across sequence dimension)
        if self.gate_type == "scalar":
            # Use scalar gates (broadcast to all features)
            # alpha: (N, 1) -> (N, 1, 1) -> (N, F, 256)
            # beta: (N, 1) -> (N, 1, 1) -> (N, F, 256)
            alpha_expanded = alpha.unsqueeze(1).expand(-1, F, -1)  # (N, F, 1)
            beta_expanded = beta.unsqueeze(1).expand(-1, F, -1)    # (N, F, 1)
            
            cnn_gated = cnn_projected * alpha_expanded  # (N, F, 256)
            handcrafted_gated = handcrafted_projected * beta_expanded   # (N, F, 256)
            xlsr_gated = xlsr_projected * 1.0  # (N, F, 256) - no gate for XLS-R
        else:
            # Use vector gates (learn separate gate for each feature)
            # This would require modifying the gate network to output (N, 256 * 2)
            raise NotImplementedError("Vector gates not yet implemented")
        
        # Sum the gated features
        fused_features = cnn_gated + handcrafted_gated + xlsr_gated  # (N, F, 256)
        
        # Apply dropout after fusion
        fused_features = self.fusion_dropout(fused_features)
        
        # Apply normalization after fusion
        fused_features = self.fusion_norm(fused_features)
        
        return fused_features
    
    def get_gate_values(self, branch_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get the gating values for analysis.
        
        Args:
            branch_features: List of feature tensors
            
        Returns:
            Dictionary containing gate values and fused features
        """
        cnn_features, handcrafted_features, xlsr_features = branch_features
        
        # Mean pool XLS-R sequence features for gate generation
        xlsr_pooled = torch.mean(xlsr_features, dim=1)  # (N, 256)
        
        # Generate gating values
        gates = self.gate_network(xlsr_pooled)
        alpha, beta = torch.split(gates, 1, dim=1)
        
        # Apply softmax to get values that sum to 1
        gates_combined = torch.cat([alpha, beta], dim=1)  # (N, 2)
        gates_softmax = torch.softmax(gates_combined, dim=1)  # (N, 2)
        alpha, beta = torch.split(gates_softmax, 1, dim=1)  # Each: (N, 1)
        
        return {
            'alpha': alpha,  # CNN gate
            'beta': beta,    # Handcrafted gate
            'xlsr_pooled': xlsr_pooled
        }
    
    def get_feature_dimensions(self) -> Dict[str, Any]:
        """Get the feature dimensions and configuration."""
        return {
            'fusion_type': 'gated',
            'branch_dims': self.branch_dims,
            'common_dim': self.common_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'gate_type': self.gate_type,
            'output_shape': f'(N, F, {self.common_dim})'
        } 