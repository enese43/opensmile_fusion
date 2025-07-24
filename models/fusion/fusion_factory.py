import sys
import os
sys.path.append(os.path.dirname(__file__))
from typing import List, Dict, Any
from gated_fusion import GatedFusion


class FusionFactory:
    """
    Factory class for creating gated fusion method.
    """
    
    @staticmethod
    def create_fusion(fusion_type: str, 
                     branch_dims: List[int],
                     **kwargs) -> GatedFusion:
        """
        Create a gated fusion module.
        
        Args:
            fusion_type: Type of fusion (only "gated" supported)
            branch_dims: List of feature dimensions from each branch
            **kwargs: Additional arguments for the fusion module
            
        Returns:
            GatedFusion module instance
        """
        
        if fusion_type.lower() == "gated":
            return GatedFusion(branch_dims, **kwargs)
        else:
            raise ValueError(f"Only 'gated' fusion type is supported. Got: {fusion_type}")
    
    @staticmethod
    def get_available_fusion_types() -> List[str]:
        """Get list of available fusion types."""
        return ["gated"]
    
    @staticmethod
    def get_fusion_info(fusion_type: str) -> Dict[str, Any]:
        """Get information about gated fusion."""
        
        if fusion_type.lower() == "gated":
            return {
                "description": "Learnable gates to dynamically weight branch contributions",
                "output_shape": "(N, F, 256)",
                "pros": ["Adaptive weighting", "Interpretable gates", "Efficient", "Maintains sequence dimension"],
                "cons": ["More complex", "Requires training"],
                "best_for": "When modalities have varying importance, interpretability needed, sequence-aware fusion"
            }
        else:
            return {"description": "Only 'gated' fusion type is supported"} 