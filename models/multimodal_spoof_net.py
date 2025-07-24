import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'fusion'))
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from models.branches.cnn_branch import CNNBranch
from models.branches.handcrafted_branch import HandcraftedBranch
from models.branches.xlsr_branch import XLSRBranch
from models.fusion.fusion_factory import FusionFactory


class MultiModalSpoofNet(nn.Module):
    """
    Multimodal Audio Spoofing Detection Network
    
    Modular architecture with three parallel branches:
    - Branch 1: Modified ResNet-18 CNN for spectrograms (4 channels)
    - Branch 2: Hand-crafted feature vector processing (OpenSMILE features)
    - Branch 3: XLS-R/Wav2Vec2.0 for raw waveform processing
    - Fusion: Gated fusion strategy for adaptive feature weighting (256-dim)
    """
    
    def __init__(self,
                 resnet_ckpt: str = "best_model_multitask_eer_refactored4_lr_adam.pth",
                 freeze_backbone: bool = True,
                 freeze_xlsr: bool = True,
                 num_classes: int = 2,
                 fusion_type: str = "gated",
                 fusion_kwargs: Optional[dict] = None):
        """
        Initialize the multimodal spoofing detection network.
        
        Args:
            resnet_ckpt: Path to pre-trained ResNet-18 weights
            freeze_backbone: Whether to freeze the ResNet backbone initially
            freeze_xlsr: Whether to freeze the XLS-R backbone initially
            num_classes: Number of output classes (not used in gated fusion)
            fusion_type: Type of fusion to use (only "gated" supported)
            fusion_kwargs: Additional arguments for the fusion module
        """
        super().__init__()
        
        # Initialize the three branches
        self.cnn_branch = CNNBranch(resnet_ckpt, freeze_backbone)
        self.handcrafted_branch = HandcraftedBranch(fixed_weight_npy="output_weights.npy")
        self.xlsr_branch = XLSRBranch(freeze_backbone=freeze_xlsr)
        
        # Get feature dimensions from each branch
        branch_dims = [
            self.cnn_branch.get_feature_dimension(),
            self.handcrafted_branch.get_feature_dimension(),
            self.xlsr_branch.get_feature_dimension()
        ]
        
        # Set default fusion kwargs
        if fusion_kwargs is None:
            fusion_kwargs = {}
        
        # Initialize fusion head using factory
        self.fusion_head = FusionFactory.create_fusion(
            fusion_type=fusion_type,
            branch_dims=branch_dims,
            num_classes=num_classes,
            **fusion_kwargs
        )
        
        # Store fusion type for reference
        self.fusion_type = fusion_type
        
    def forward(self, 
                spectrogram: torch.Tensor,
                handcrafted_features: torch.Tensor,
                waveform: torch.Tensor,
                return_embeddings: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through the multimodal network.
        
        Args:
            spectrogram: Spectrogram input (N, 4, H, W) - 4 channels for your model
            handcrafted_features: Hand-crafted feature vector (N, 6373)
            waveform: Raw waveform input (N, L) at 16kHz
            return_embeddings: Whether to return branch embeddings
            
        Returns:
            logits: Classification logits (N, 2)
            embeddings: Optional tuple of branch embeddings
        """
        
        # Get sequence length from XLS-R branch
        sequence_length = self.xlsr_branch.get_sequence_length(waveform)
        
        # Process through each branch
        cnn_features = self.cnn_branch(spectrogram, sequence_length)      # (N, F, 256)
        handcrafted_features = self.handcrafted_branch(handcrafted_features, sequence_length)  # (N, F, 256)
        xlsr_features = self.xlsr_branch(waveform)                        # (N, F, 256)
        
        # Combine features through gated fusion head
        branch_features = [cnn_features, handcrafted_features, xlsr_features]
        fused_features = self.fusion_head(branch_features)  # (N, F, 256)
        
        # Note: This model doesn't output logits directly since it's used in the pipeline
        # The fused features are passed to AASIST backend for final classification
        
        if return_embeddings:
            embeddings = (cnn_features, handcrafted_features, xlsr_features)
            return fused_features, embeddings
        else:
            return fused_features
    
    def freeze_backbone(self):
        """Freeze the ResNet backbone parameters."""
        self.cnn_branch.set_backbone_frozen(True)
    
    def unfreeze_backbone(self):
        """Unfreeze the ResNet backbone parameters."""
        self.cnn_branch.set_backbone_frozen(False)
    
    def set_xlsr_frozen(self, frozen: bool):
        """Set whether the XLS-R backbone should be frozen."""
        self.xlsr_branch.set_backbone_frozen(frozen)
    
    def get_feature_dimensions(self) -> dict:
        """Get the feature dimensions for each component."""
        fusion_dims = self.fusion_head.get_feature_dimensions()
        return {
            'cnn_branch': self.cnn_branch.get_feature_dimension(),
            'handcrafted_branch': self.handcrafted_branch.get_feature_dimension(),
            'xlsr_branch': self.xlsr_branch.get_feature_dimension(),
            'fusion_head': fusion_dims,
            'fusion_type': self.fusion_type
        }
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get the learned feature importance weights from the handcrafted branch."""
        return self.handcrafted_branch.get_feature_importance()
    
    def get_gate_values(self, spectrogram: torch.Tensor, 
                       handcrafted_features: torch.Tensor, 
                       waveform: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get gate values from gated fusion.
        
        Args:
            spectrogram: Spectrogram input
            handcrafted_features: Hand-crafted feature vector
            waveform: Raw waveform input
            
        Returns:
            Gate values dictionary or None if not using gated fusion
        """
        if hasattr(self.fusion_head, 'get_gate_values'):
            sequence_length = self.xlsr_branch.get_sequence_length(waveform)
            cnn_features = self.cnn_branch(spectrogram, sequence_length)
            handcrafted_features = self.handcrafted_branch(handcrafted_features, sequence_length)
            xlsr_features = self.xlsr_branch(waveform)
            branch_features = [cnn_features, handcrafted_features, xlsr_features]
            return self.fusion_head.get_gate_values(branch_features)
        return None


# Example usage and testing
if __name__ == "__main__":
    # Create model instance with gated fusion
    model = MultiModalSpoofNet(
        freeze_backbone=True, 
        freeze_xlsr=True,
        fusion_type="gated",
        fusion_kwargs={}
    )
    
    # Create dummy inputs
    batch_size = 4
    spectrogram = torch.randn(batch_size, 4, 64, 64)  # (N, 4, H, W) - 4 channels
    handcrafted_features = torch.randn(batch_size, 6373)  # (N, 6373)
    waveform = torch.randn(batch_size, 16000)  # (N, L) - 1 second at 16kHz
    
    # Forward pass
    fused_features, embeddings = model(spectrogram, handcrafted_features, waveform, return_embeddings=True)
    
    print(f"Input shapes:")
    print(f"  Spectrogram: {spectrogram.shape}")
    print(f"  Hand-crafted features: {handcrafted_features.shape}")
    print(f"  Waveform: {waveform.shape}")
    print(f"\nOutput shapes:")
    print(f"  Fused features: {fused_features.shape}")
    print(f"  CNN embeddings: {embeddings[0].shape}")
    print(f"  Hand-crafted embeddings: {embeddings[1].shape}")
    print(f"  XLS-R embeddings: {embeddings[2].shape}")
    
    # Test feature dimensions
    dims = model.get_feature_dimensions()
    print(f"\nFeature dimensions: {dims}")
    
    # Test fusion factory
    print(f"\nAvailable fusion types: {FusionFactory.get_available_fusion_types()}")
    fusion_info = FusionFactory.get_fusion_info("gated")
    print(f"Gated fusion info: {fusion_info}") 