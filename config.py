"""
Configuration file for Fusion + AASIST Audio Spoofing Detection Network
"""

import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Model paths
    resnet_ckpt: str = "best_model_multitask_eer_refactored4_lr_adam.pth"
    fusion_type: str = "gated"
    fusion_kwargs: dict = None
    
    # Model parameters
    freeze_backbone: bool = False  # Freeze CNN backbone
    freeze_xlsr: bool = False  # Freeze XLS-R backbone
    num_classes: int = 2
    
    def __post_init__(self):
        if self.fusion_kwargs is None:
            # Gated fusion doesn't need pooling_method
            self.fusion_kwargs = {}


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    # Data paths for ASVspoof2019 LA dataset
    # Spectrograms (Gabor features)
    spectrogram_train: str = "/ds-slt/audio/ASVSpoof_LA_19/formant_gabor_feat/gabor_train_npy"
    spectrogram_dev: str = "/ds-slt/audio/ASVSpoof_LA_19/formant_gabor_feat/gabor_dev_npy"
    spectrogram_test: str = "/ds-slt/audio/ASVSpoof_LA_19/formant_gabor_feat/gabor_eval_npy"
    
    # Handcrafted features (OpenSMILE)
    features_train: str = "/ds-slt/audio/ASVSpoof_LA_19/opensmile_LA_compare/train"
    features_dev: str = "/ds-slt/audio/ASVSpoof_LA_19/opensmile_LA_compare/dev"
    features_test: str = "/ds-slt/audio/ASVSpoof_LA_19/opensmile_LA_compare/eval"
    
    # Audio files (FLAC)
    audio_train: str = "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_train/flac"
    audio_dev: str = "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_dev/flac"
    audio_test: str = "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_eval/flac"
    
    # Dataset parameters
    use_real_labels: bool = True
    sample_rate: int = 16000
    seed: int = 42
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 0
    drop_last: bool = True
    
    # Audio parameters
    max_audio_length: int = 96000  # 6 seconds at 16kHz


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training parameters
    num_epochs: int = 30
    learning_rate: float = 1e-6
    weight_decay: float = 1e-4  # Increased from 1e-4 to 1e-3 for stronger regularization
    
    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "plateau"  # "cosine", "step", "plateau"
    
    # Loss
    loss_function: str = "cross_entropy"
    class_weights: list = None  # Class weights for imbalanced dataset
    
    # Regularization
    dropout_rate: float = 0.2  # Add dropout to prevent overfitting
    early_stopping_patience: int = 15  # Stop training if validation doesn't improve
    gradient_clip_norm: float = None  # Gradient clipping
    
    # Training settings
    device: str = None  # Will be set to "cuda" if available, else "cpu"
    save_dir: str = "checkpoints"
    save_freq: int = 1  # Save every N epochs
    
    # Wandb settings
    use_wandb: bool = True
    project_name: str = "fusion-aasist"
    run_name: str = "30_epoch_unfrozen_weights19_wd1e3_dropout02"  # Updated run name
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.class_weights is None:
            # Class weights: [spoof, bonafide] - inverse frequency weighting
            self.class_weights = [1.0, 1.0]


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    
    # Evaluation settings
    batch_size: int = 32
    num_workers: int = 0
    
    # Metrics
    metrics: list = None
    
    # Wandb settings
    use_wandb: bool = True
    project_name: str = "fusion-aasist-eval"
    run_name: str = "evaluation"  # Wandb run name for evaluation
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc", "eer"]


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    def print_config(self):
        """Print the current configuration."""
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        
        print("MODEL CONFIG:")
        print(f"  ResNet checkpoint: {self.model.resnet_ckpt}")
        print(f"  Fusion type: {self.model.fusion_type}")
        print(f"  Fusion kwargs: {self.model.fusion_kwargs}")
        print(f"  Freeze backbone: {self.model.freeze_backbone}")
        print(f"  Freeze XLS-R: {self.model.freeze_xlsr}")
        print(f"  Num classes: {self.model.num_classes}")
        
        print("\nDATA CONFIG:")
        print(f"  Spectrogram train: {self.data.spectrogram_train}")
        print(f"  Spectrogram dev: {self.data.spectrogram_dev}")
        print(f"  Spectrogram test: {self.data.spectrogram_test}")
        print(f"  Features train: {self.data.features_train}")
        print(f"  Features dev: {self.data.features_dev}")
        print(f"  Features test: {self.data.features_test}")
        print(f"  Audio train: {self.data.audio_train}")
        print(f"  Audio dev: {self.data.audio_dev}")
        print(f"  Audio test: {self.data.audio_test}")
        print(f"  Use real labels: {self.data.use_real_labels}")
        print(f"  Sample rate: {self.data.sample_rate}")
        print(f"  Max audio length: {self.data.max_audio_length} samples ({self.data.max_audio_length/16000:.1f}s)")
        print(f"  Batch size: {self.data.batch_size}")
        print(f"  Seed: {self.data.seed}")
        
        print("\nTRAINING CONFIG:")
        print(f"  Num epochs: {self.training.num_epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Weight decay: {self.training.weight_decay}")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Scheduler: {self.training.scheduler}")
        print(f"  Loss function: {self.training.loss_function}")
        print(f"  Class weights: {self.training.class_weights}")
        print(f"  Device: {self.training.device}")
        print(f"  Save directory: {self.training.save_dir}")
        print(f"  Use wandb: {self.training.use_wandb}")
        print(f"  Project name: {self.training.project_name}")
        print(f"  Run name: {self.training.run_name}")
        
        print("\nEVALUATION CONFIG:")
        print(f"  Batch size: {self.evaluation.batch_size}")
        print(f"  Use wandb: {self.evaluation.use_wandb}")
        print(f"  Project name: {self.evaluation.project_name}")
        print(f"  Run name: {self.evaluation.run_name}")
        print(f"  Metrics: {self.evaluation.metrics}")
        print("="*60)


# Default configuration
default_config = Config()


if __name__ == "__main__":
    # Test the configuration
    config = default_config
    config.print_config() 