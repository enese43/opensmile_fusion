# Fusion + AASIST Audio Spoofing Detection Network

A PyTorch implementation of a multimodal audio spoofing detection system that combines a fusion model with the AASIST backend for enhanced performance.

## Features

- **Fusion + AASIST Architecture**: Combines multimodal fusion features with AASIST backend
- **Advanced Fusion**: Gated fusion system for adaptive feature weighting
- **Complete Pipeline**: Training and evaluation in one command
- **Flexible Configuration**: Easy-to-use configuration system with JSON files and command-line overrides
- **Wandb Integration**: Comprehensive logging and experiment tracking
- **EER Evaluation**: Equal Error Rate calculation for spoofing detection

## Quick Start

### Complete Pipeline (Training + Evaluation)

Run the entire pipeline from training to evaluation:

```bash
# Using default configuration
python main.py

# Override specific parameters
python main.py --num_epochs 10 --batch_size 4 --data_root "custom_data"

# Quick test (2 epochs, no wandb)
python main.py --num_epochs 2 --batch_size 1 --no_wandb

# Full training (50 epochs, wandb logging)
python main.py --num_epochs 50 --batch_size 8 --learning_rate 5e-5
```

## Data Structure

Your data should be organized as follows:

```
data/
├── train/
│   ├── audio/           # Audio files (.wav, .flac, .mp3)
│   ├── spectrograms/    # Spectrogram files (.npy)
│   ├── features/        # OpenSMILE features (.npy)
│   └── metadata.json    # Optional: sample metadata
├── val/
│   ├── audio/
│   ├── spectrograms/
│   ├── features/
│   └── metadata.json
└── test/
    ├── audio/
    ├── spectrograms/
    ├── features/
    └── metadata.json
```

## Configuration System

The system uses a simple dataclass-based configuration system with command-line overrides.

### Configuration Structure

The configuration is organized into four main sections:

- **Model**: Architecture settings, checkpoint paths, fusion type
- **Data**: Data paths, batch sizes, audio parameters
- **Training**: Epochs, learning rate, optimizer, wandb settings
- **Evaluation**: Evaluation batch size, metrics, wandb settings

### Using Configuration

#### Default Configuration
```bash
python main.py  # Uses default configuration
```

#### Print Configuration
```bash
python main.py --print_config
```

### Command-Line Overrides

You can override any configuration parameter via command line:

```bash
# Override data paths
python main.py --data_root "custom_data" --train_split "train_custom"

# Override training parameters
python main.py --num_epochs 20 --batch_size 8 --learning_rate 5e-5

# Override model parameters
python main.py --resnet_ckpt "custom_model.pth" --fusion_type "gated"

# Disable wandb
python main.py --no_wandb
```

### Default Configuration Values

```python
# Model Configuration
resnet_ckpt = "best_model_multitask_eer_refactored4_lr_adam.pth"
fusion_type = "gated"
freeze_backbone = True
freeze_xlsr = True

# Data Configuration
data_root = "data"
batch_size = 2
use_real_labels = True

# Training Configuration
num_epochs = 5
learning_rate = 1e-4
device = "cuda" if available, else "cpu"
use_wandb = True
```

## Model Architecture

### Fusion + AASIST Pipeline

The system combines a multimodal fusion model with the AASIST backend:

1. **Fusion Model**: Combines spectrogram, handcrafted features, and audio waveforms
2. **AASIST Backend**: Processes fusion features for final spoofing detection

### Fusion Methods

- **Gated Fusion**: Learnable gates for adaptive feature weighting and feature importance control

### Data Processing

- **Spectrograms**: 4-channel spectrogram features
- **Handcrafted Features**: OpenSMILE-based features
- **Audio Waveforms**: Raw audio processed through XLS-R model

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Training and Evaluation

```bash
# Complete pipeline with default config
python main.py

# Quick test (2 epochs, no wandb)
python main.py --num_epochs 2 --batch_size 1 --no_wandb

# Full training (50 epochs, wandb logging)
python main.py --num_epochs 50 --batch_size 8 --learning_rate 5e-5
```

### Advanced Usage

```bash
# Custom parameters
python main.py --num_epochs 100 --batch_size 16 --learning_rate 1e-5

# Different data paths
python main.py --data_root "custom_data" --train_split "train_custom"

# GPU training with custom learning rate
python main.py --device "cuda" --learning_rate 1e-5

# Print current configuration
python main.py --print_config
```

## Output

The system generates:

- **Checkpoints**: `checkpoints/fused_assist_model.pth`
- **Wandb logs**: Comprehensive experiment tracking and metrics
- **Final results**: `final_results.json` with all evaluation metrics
- **Configuration files**: JSON configs for experiment reproducibility

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- AUC (Area Under Curve)
- EER (Equal Error Rate)
- Confusion Matrix

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchaudio
- transformers
- librosa
- scikit-learn
- numpy
- wandb (optional, for experiment tracking)

## License

[Your License Here] # opensmile_fusion
