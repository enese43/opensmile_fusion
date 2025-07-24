import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Constants from the original model
INPUT_CHANNELS = 4
NUM_ATTACK_TYPES = 20

class ResidualBlock(nn.Module):
    """Standard Residual Block for ResNet."""
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # Shortcut connection to match dimensions if needed
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ModifiedResNet(nn.Module):
    """Modified ResNet that uses only one residual layer and outputs 128 features using a CNN."""
    def __init__(self, block, input_channels: int = INPUT_CHANNELS):
        super(ModifiedResNet, self).__init__()
        self.inchannel = 64

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Only one residual layer (layer1 from original)
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        
        # CNN to get 128 channels
        self.cnn_to_128 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_128 = nn.BatchNorm2d(128)
        self.relu_128 = nn.ReLU(inplace=True)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def make_layer(self, block, channels, num_blocks, stride):
        """Creates a ResNet layer composed of multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.inchannel, channels, stride_val))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Pass through initial conv and one residual layer
        out = self.conv1(x)
        out = self.layer1(out)
        
        # Pass through 128 channel layer
        out = self.cnn_to_128(out)
        out = self.bn_128(out)
        out = self.relu_128(out)
        
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)  # Flatten: [batch_size, 128]
        return features

class CNNBranch(nn.Module):
    """
    Branch 1: Modified ResNet-18 CNN for spectrogram processing
    
    Architecture:
    - Modified ResNet-18 (only one residual layer)
    - 1x1 Conv2d to get 128 channels
    - Global Average Pooling
    - BatchNorm for normalization
    - Output: (N, 128) which gets broadcast to (N, F, 128)
    """
    
    def __init__(self, 
                 resnet_ckpt: Optional[str] = None,
                 freeze_backbone: bool = True,
                 input_channels: int = INPUT_CHANNELS,
                 output_dim: int = 128):
        """
        Initialize the CNN branch.
        
        Args:
            resnet_ckpt: Path to pre-trained ResNet weights (optional)
            freeze_backbone: Whether to freeze the ResNet backbone
            input_channels: Number of input channels
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # Create the modified ResNet backbone
        self.backbone = ModifiedResNet(ResidualBlock, input_channels=input_channels)
        
        # Add BatchNorm for normalization
        self.batch_norm = nn.BatchNorm1d(output_dim)
        
        # Load pre-trained weights if provided
        if resnet_ckpt is not None:
            self.load_weights_from_checkpoint(resnet_ckpt)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def load_weights_from_checkpoint(self, checkpoint_path: str):
        """Load weights from checkpoint, mapping only the compatible layers."""
        print(f"Loading CNN weights from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create a mapping of layer names
        model_dict = self.backbone.state_dict()
        loaded_layers = 0
        total_layers = len(model_dict)
        
        for name, param in model_dict.items():
            if name in checkpoint:
                # Direct match
                model_dict[name] = checkpoint[name]
                loaded_layers += 1
                print(f"✓ Loaded: {name}")
            else:
                print(f"✗ Not found: {name}")
        
        # Load the weights
        self.backbone.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {loaded_layers}/{total_layers} layers from checkpoint")
    
    def _freeze_backbone(self):
        """Freeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def set_backbone_frozen(self, frozen: bool):
        """Set whether the backbone should be frozen."""
        if frozen:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
        self.freeze_backbone = frozen
    
    def forward(self, spectrogram: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through the CNN branch.
        
        Args:
            spectrogram: Spectrogram input (N, C, H, W)
            sequence_length: Optional sequence length F for broadcasting
            
        Returns:
            CNN features (N, 128) or (N, F, 128) if sequence_length is provided
        """
        # Process through backbone
        cnn_features = self.backbone(spectrogram)  # (N, 128)
        
        # Apply BatchNorm normalization
        cnn_features = self.batch_norm(cnn_features)  # (N, 128)
        
        # Broadcast to sequence length if provided
        if sequence_length is not None:
            cnn_features = cnn_features.unsqueeze(1).repeat(1, sequence_length, 1)  # (N, F, 128)
        
        return cnn_features
    
    def get_feature_dimension(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim 