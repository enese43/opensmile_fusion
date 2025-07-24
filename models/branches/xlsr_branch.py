import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import Optional


class XLSRBranch(nn.Module):
    """
    Branch 3: XLS-R/Wav2Vec2.0 for raw waveform processing
    
    Architecture:
    - HuggingFace Wav2Vec2-XLS-R-300M (feature size = 1024)
    - Linear(1024, 128) + LayerNorm + GELU
    - Output: (N, F, 128) where F is the sequence length
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-xls-r-300m",
                 input_dim: int = 1024,
                 output_dim: int = 128,
                 freeze_backbone: bool = True,
                 expected_sample_rate: int = 16000):  # Add this
        super().__init__()
        self.expected_sample_rate = expected_sample_rate
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained XLS-R model
        self.xlsr_model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze XLS-R backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),    # Project from 1024 to 128 dimensions
            nn.LayerNorm(output_dim),            # Layer normalization
            nn.GELU()                            # GELU activation
        )
    
    def _freeze_backbone(self):
        """Freeze the XLS-R backbone parameters."""
        for param in self.xlsr_model.parameters():
            param.requires_grad = False
        self.xlsr_model.eval()  # Add this

    def _unfreeze_backbone(self):
        """Unfreeze the XLS-R backbone parameters."""
        for param in self.xlsr_model.parameters():
            param.requires_grad = True
        self.xlsr_model.train()  # Add this
    def set_backbone_frozen(self, frozen: bool):
        """Set whether the backbone should be frozen."""
        if frozen:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
        self.freeze_backbone = frozen
    
    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the XLS-R branch.
        
        Args:
            waveform: Raw waveform input (N, L) at 16kHz
            attention_mask: Optional attention mask for variable length inputs
            
        Returns:
            XLS-R features (N, F, 128) where F is the sequence length
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(waveform)
        
        # Forward through XLS-R model
        if self.freeze_backbone:
            self.xlsr_model.eval()  # Ensure eval mode
            with torch.no_grad():
                xlsr_outputs = self.xlsr_model(waveform, attention_mask=attention_mask)
        else:
            xlsr_outputs = self.xlsr_model(waveform, attention_mask=attention_mask)
        
        # Extract hidden states
        xlsr_features = xlsr_outputs.last_hidden_state  # (N, F, 1024)
        
        # Project through custom layers
        xlsr_projected = self.projection(xlsr_features)  # (N, F, 128)
        
        return xlsr_projected
    
    def get_feature_dimension(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim
    
    def get_sequence_length(self, waveform: torch.Tensor) -> int:
        """
        Get the sequence length F from the XLS-R model.
        
        Args:
            waveform: Raw waveform input (N, L)
            
        Returns:
            Sequence length F
        """
        with torch.no_grad():
            xlsr_outputs = self.xlsr_model(waveform)
            return xlsr_outputs.last_hidden_state.size(1) 