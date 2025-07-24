import torch
import torch.nn as nn
import sys
import os
from typing import Optional

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/fusion'))

from models.multimodal_spoof_net import MultiModalSpoofNet
from assist import Model as AASISTModel


class FusionAASISTPipeline(nn.Module):
    """
    Complete pipeline that combines fusion model with AASIST backend
    
    Flow:
    1. Extract features from three branches (CNN, Handcrafted, XLS-R)
    2. Fuse features using gated fusion
    3. Feed fused features to AASIST backend for final classification
    """
    
    def __init__(self, 
                 resnet_ckpt: str = "best_model_multitask_eer_refactored4_lr_adam.pth",
                 freeze_backbone: bool = True,
                 freeze_xlsr: bool = True,
                 fusion_type: str = "gated",
                 fusion_kwargs: Optional[dict] = None,
                 device: str = 'cuda'):
        """
        Initialize the complete pipeline.
        
        Args:
            resnet_ckpt: Path to pre-trained ResNet-18 weights
            freeze_backbone: Whether to freeze the ResNet backbone
            freeze_xlsr: Whether to freeze the XLS-R backbone
            fusion_type: Type of fusion to use (now using "gated")
            fusion_kwargs: Additional arguments for fusion
            device: Device to run the model on
        """
        super().__init__()
        
        self.device = device
        
        # Initialize the fusion model (without final classification)
        self.fusion_model = MultiModalSpoofNet(
            resnet_ckpt=resnet_ckpt,
            freeze_backbone=freeze_backbone,
            freeze_xlsr=freeze_xlsr,
            num_classes=2,  # This will be ignored since we're not using the final classifier
            fusion_type=fusion_type,
            fusion_kwargs=fusion_kwargs
        )
        
        # Get the fusion head to extract intermediate features
        self.fusion_head = self.fusion_model.fusion_head
        
        # Initialize AASIST backend
        class MockArgs:
            pass
        args = MockArgs()
        self.aasist_backend = AASISTModel(args, device)
        
        # Move everything to device
        self.to(device)
        
    def forward(self, 
                spectrogram: torch.Tensor,
                handcrafted_features: torch.Tensor,
                waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete pipeline.
        
        Args:
            spectrogram: Spectrogram input (N, 4, H, W)
            handcrafted_features: Hand-crafted feature vector (N, 6373)
            waveform: Raw waveform input (N, L) at 16kHz
            
        Returns:
            Final classification logits (N, 2)
        """
        
        # Step 1: Extract features from each branch
        xlsr_features = self.fusion_model.xlsr_branch(waveform)  # (N, F, 256)
        sequence_length = xlsr_features.shape[1]
        
        cnn_features = self.fusion_model.cnn_branch(spectrogram, sequence_length)      # (N, F, 256)
        handcrafted_features = self.fusion_model.handcrafted_branch(handcrafted_features, sequence_length)  # (N, F, 256)
        
        # Step 2: Fuse features using GATED FUSION
        branch_features = [cnn_features, handcrafted_features, xlsr_features]
        fused_features = self.fusion_head(branch_features)  # (N, F, 256) - gated fusion output
        
        # Step 3: Pass through AASIST backend
        final_output = self.aasist_backend(fused_features)  # (N, 2)
        
        return final_output
    
    def get_feature_dimensions(self) -> dict:
        """Get the feature dimensions for each component."""
        fusion_dims = self.fusion_model.get_feature_dimensions()
        return {
            'fusion_model': fusion_dims,
            'aasist_backend': 'Expected input: [N, F, 256]',
            'final_output': '[N, 2]',
            'fusion_method': 'gated fusion',
            'feature_dimension': '256 (all branches)'
        }


def test_complete_pipeline():
    """Test the complete fusion + AASIST pipeline."""
    
    print("=== Testing Complete Fusion + AASIST Pipeline ===")
    print("Fusion Method: GATED FUSION")
    print("All branches output: [N, F, 256]")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create pipeline
    pipeline = FusionAASISTPipeline(
        resnet_ckpt="best_model_multitask_eer_refactored4_lr_adam.pth",
        freeze_backbone=True,
        freeze_xlsr=True,
        fusion_type="gated",
        fusion_kwargs={},  # Gated fusion doesn't need pooling_method
        device=device
    )
    
    # Create dummy inputs
    batch_size = 32
    spectrogram = torch.randn(batch_size, 4, 64, 64).to(device)  # (N, 4, H, W)
    handcrafted_features = torch.randn(batch_size, 6373).to(device)  # (N, 6373)
    waveform = torch.randn(batch_size, 96000).to(device)  # (N, L) - 6 seconds at 16kHz
    
    print(f"\nInput shapes:")
    print(f"  Spectrogram: {spectrogram.shape}")
    print(f"  Hand-crafted features: {handcrafted_features.shape}")
    print(f"  Waveform: {waveform.shape}")
    
    # Forward pass
    pipeline.eval()
    with torch.no_grad():
        # Use the pipeline's forward method (this is the correct way)
        output = pipeline(spectrogram, handcrafted_features, waveform)
        
        # For debugging, also get intermediate features
        xlsr_features = pipeline.fusion_model.xlsr_branch(waveform)
        print(f"\nXLSR output shape: {xlsr_features.shape}")
        
        sequence_length = xlsr_features.shape[1]
        cnn_features = pipeline.fusion_model.cnn_branch(spectrogram, sequence_length)
        handcrafted_features_ = pipeline.fusion_model.handcrafted_branch(handcrafted_features, sequence_length)
        
        # Get fused feature shape
        branch_features = [cnn_features, handcrafted_features_, xlsr_features]
        fused_features = pipeline.fusion_head(branch_features)
        print(f"Fused feature shape (input to AASIST): {fused_features.shape}")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{batch_size}, 2]")
    
    # Test feature dimensions
    dims = pipeline.get_feature_dimensions()
    print(f"\nFeature dimensions: {dims}")
    
    print("\n=== Pipeline Test Completed Successfully! ===")
    return pipeline, output


if __name__ == "__main__":
    # Run the test
    pipeline, output = test_complete_pipeline() 