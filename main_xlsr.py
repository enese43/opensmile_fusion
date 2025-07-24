#!/usr/bin/env python3
"""
Main entry point for XLS-R + AASIST Audio Spoofing Detection Network
Two experiments:
1. Frozen backbone: XLS-R (frozen) + AASIST (random) → Evaluation only
2. Training: XLS-R (trainable) + AASIST (trainable) → Full training
"""

import argparse
import json
import os
import torch
import sys
import wandb
import numpy as np
import librosa
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, f1_score
from models.branches.xlsr_branch_2 import XLSRBranch2

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/fusion'))

from main_fused_assist import FusionAASISTPipeline
from config import default_config


def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER) for anti-spoofing.
    
    Args:
        labels: True labels (0=spoof, 1=bonafide)
        scores: Spoof scores (higher score = more likely spoof)
    
    Returns:
        eer: Equal Error Rate
        threshold: EER threshold
    """
    # Convert inputs to numpy arrays
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Check for edge cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class present in labels: {unique_labels}")
        return 1.0, 0.0  # Return 100% EER for single-class case
    
    # For anti-spoofing: 
    # - Bonafide (1) should be rejected when score > threshold (False Rejection)
    # - Spoof (0) should be accepted when score < threshold (False Acceptance)
    
    # Invert labels for ROC calculation: 0->1 (spoof as positive), 1->0 (bonafide as negative)
    # This makes ROC curve interpret higher scores as positive class (spoof detection)
    inverted_labels = 1 - labels
    
    try:
        fpr, tpr, thresholds = roc_curve(inverted_labels, scores)
        
        # In this setup:
        # FPR = False Positive Rate = Bonafide incorrectly classified as Spoof = FRR
        # TPR = True Positive Rate = Spoof correctly classified as Spoof = 1 - FAR
        frr = fpr      # False Rejection Rate (bonafide rejected)
        far = 1 - tpr   # False Acceptance Rate (spoof accepted)
        
        # Find threshold where FRR = FAR
        abs_diff = np.abs(frr - far)
        min_idx = np.argmin(abs_diff)
        
        eer = (frr[min_idx] + far[min_idx]) / 2
        threshold = thresholds[min_idx]
        
        # Validate EER
        if np.isnan(eer) or np.isinf(eer):
            return 1.0, 0.0
        
        return eer, threshold
        
    except Exception as e:
        print(f"Error in ROC calculation: {e}")
        return 1.0, 0.0


def custom_collate_fn(batch):
    """
    Custom collate function for XLS-R + AASIST (audio only).
    Handles variable-length waveforms by padding/truncating to fixed length.
    """
    waveforms = []
    labels = []
    sample_ids = []
    
    for item in batch:
        waveforms.append(item['waveform'])
        labels.append(item['label'])
        sample_ids.append(item['sample_id'])
    
    # Stack tensors
    return {
        'waveform': torch.stack(waveforms),
        'label': torch.tensor(labels),
        'sample_id': sample_ids
    }


class XLSRAASISTModel(torch.nn.Module):
    def __init__(self, freeze_backbone=True, device='cuda'):
        super().__init__()
        self.device = device
        self.freeze_backbone = freeze_backbone
        
        # Import AASIST model
        from assist import Model as AASISTModel
        
        # Initialize AASIST backend
        class MockArgs:
            pass
        args = MockArgs()
        self.aasist_backend = AASISTModel(args, device)
        
        # Import and create XLS-R branch
        self.xlsr_branch = XLSRBranch2(freeze_backbone=freeze_backbone)
        
        # Add feature dimension adapter: 1024 → 126 
        # (so after AASIST's //3 pooling: 126//3 = 42 to match pos_S)
        self.feature_adapter = nn.Sequential(
            nn.Linear(1024, 256),  # Map from XLS-R's 1024 to 126 features
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Remove sequence pooling - not needed for this issue
        
        # Move to device
        self.to(device)
    
    def forward(self, waveform):
        # Extract XLS-R features
        xlsr_features = self.xlsr_branch(waveform)  # (N, F, 1024)
        
        # Adapt feature dimensions: 1024 → 126
        # So AASIST's max_pool2d will give: 126//3 = 42 (matches pos_S)
        adapted_features = self.feature_adapter(xlsr_features)  # (N, F, 126)
        
        # Feed to AASIST - feature dimension will pool to 42
        final_output = self.aasist_backend(adapted_features)
        
        return final_output


def train_model(model, train_loader, val_loader, test_loader, device, config, run_name):
    """Train the XLS-R + AASIST model."""
    print(f"\n🚀 Starting training for {run_name}...")
    
    # Setup optimizer
    if config.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    
    # Setup scheduler
    if config.training.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    elif config.training.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_step_size, gamma=config.training.lr_gamma)
    else:
        scheduler = None
    
    # Setup loss function
    if config.training.class_weights is not None:
        class_weights_tensor = torch.tensor(config.training.class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_eer = float('inf')  # Lower EER is better
    best_model_state = None
    
    for epoch in range(config.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Train]")
        for batch in train_pbar:
            waveforms = batch['waveform'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Monitor gradient flow (optional - can remove if too verbose)
            if epoch == 0 and train_total < 100:  # Only log first epoch, first few batches
                total_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        if 'feature_adapter' in name or 'aasist_backend' in name:
                            print(f"  Grad norm {name}: {param_norm:.6f}")
                total_norm = total_norm ** (1. / 2)
                if train_total % 20 == 0:
                    print(f"  Total gradient norm: {total_norm:.6f}")
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predictions = logits.max(1)
            train_correct += predictions.eq(labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        val_scores = []  # Add scores collection here
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Val]")
            for batch in val_pbar:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(waveforms)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predictions = logits.max(1)
                val_correct += predictions.eq(labels).sum().item()
                val_total += labels.size(0)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_scores.extend(logits[:, 0].cpu().numpy())  # Collect scores here
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*val_correct/val_total:.2f}%"
                })
        
        # Calculate validation metrics
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        val_f1 = f1_score(val_labels, val_predictions, average='binary') * 100
        
        # Calculate validation EER (using already collected data)
        try:
            val_eer, val_eer_threshold = calculate_eer(val_labels, val_scores)
            # Handle single-class prediction case
            if np.isinf(val_eer) or np.isnan(val_eer):
                val_eer = 1.0  # Set to 100% EER for single-class predictions
        except Exception as e:
            print(f"Warning: Could not calculate validation EER: {e}")
            val_eer = 1.0  # Set to 100% EER if calculation fails
        
        # Test set evaluation after every epoch
        print(f"Evaluating on test set for epoch {epoch+1}...")
        test_metrics = evaluate_model(model, test_loader, device, config.training.class_weights)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log to wandb (including test metrics)
        wandb.log({
            f"{run_name}/epoch": epoch + 1,
            f"{run_name}/train_loss": train_loss / len(train_loader),
            f"{run_name}/train_accuracy": train_accuracy,
            f"{run_name}/val_loss": val_loss / len(val_loader),
            f"{run_name}/val_accuracy": val_accuracy,
            f"{run_name}/val_f1": val_f1,
            f"{run_name}/val_eer": val_eer * 100,
            f"{run_name}/test_accuracy": test_metrics['accuracy'],
            f"{run_name}/test_f1": test_metrics['f1'],
            f"{run_name}/test_eer": test_metrics['eer'],
            f"{run_name}/learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_accuracy:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, "
              f"Val Acc={val_accuracy:.2f}%, Val F1={val_f1:.2f}%, Val EER={val_eer*100:.2f}%")
        print(f"  Test Acc={test_metrics['accuracy']:.2f}%, Test F1={test_metrics['f1']:.2f}%, Test EER={test_metrics['eer']:.2f}%")
        
        # Debug: Check validation predictions distribution
        val_pred_spoof = sum(1 for pred in val_predictions if pred == 0)
        val_pred_bonafide = sum(1 for pred in val_predictions if pred == 1)
        print(f"  Val Predictions: Spoof={val_pred_spoof}, Bonafide={val_pred_bonafide}")
        
        # Save best model based on validation EER (lower is better)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_model_state = model.state_dict().copy()
            print(f"  🎯 New best model! Val EER: {val_eer*100:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  💾 Loaded best model with Val EER: {best_val_eer*100:.2f}%")
    
    return model


def evaluate_model(model, dataloader, device, class_weights=None):
    """Evaluate XLS-R + AASIST model on a dataloader."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_scores = []
    total_loss = 0.0
    
    # Initialize loss function with class weights
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            waveforms = batch['waveform'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(waveforms)
            loss = criterion(logits, labels)
            
            probabilities = torch.softmax(logits, dim=1)
            _, predictions = logits.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Prob of bonafide
            all_scores.extend(logits[:, 0].cpu().numpy())  # Use score for class 0 (spoof) for EER
            total_loss += loss.item()
    
    # Calculate confusion matrix FIRST
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Check if we have predictions for both classes
    unique_preds = np.unique(all_predictions)
    if len(unique_preds) == 1:
        # Model is predicting only one class
        print(f"⚠️  WARNING: Model predicting only class {unique_preds[0]} (0=spoof, 1=bonafide)")
        precision = recall = f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, 
            average='binary',
            pos_label=1,  # Explicitly set bonafide as positive class
            zero_division=0  # Handle edge cases
        )
    
    try:
        # AUC uses bonafide probabilities (higher prob = more likely bonafide)
        # This is correct for binary classification where bonafide=1 is positive class
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0
    
    try:
        eer, eer_threshold = calculate_eer(all_labels, all_scores)
    except Exception as e:
        eer, eer_threshold = 0.0, 0.0
    
    # Minimal debug output - only show if there's an issue
    if f1 == 0 or len(unique_preds) == 1:
        print(f"DEBUG - Confusion Matrix:")
        print(f"  TN (Spoof->Spoof): {cm[0,0]}, FP (Spoof->Bonafide): {cm[0,1]}")
        print(f"  FN (Bonafide->Spoof): {cm[1,0]}, TP (Bonafide->Bonafide): {cm[1,1]}")
        print(f"DEBUG - True Label distribution: Spoof={np.sum(np.array(all_labels)==0)}, Bonafide={np.sum(np.array(all_labels)==1)}")
        print(f"DEBUG - Predicted Label distribution: Spoof={np.sum(np.array(all_predictions)==0)}, Bonafide={np.sum(np.array(all_predictions)==1)}")
        print(f"DEBUG - Score range: Min={np.min(all_scores):.3f}, Max={np.max(all_scores):.3f}")
        print(f"DEBUG - Using spoof scores (logits[:, 0]) for EER calculation")
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'eer': eer * 100,
        'eer_threshold': eer_threshold,
        'loss': total_loss / len(dataloader),
        'confusion_matrix': cm
    }


class DirectDataset(torch.utils.data.Dataset):
    """
    Direct dataset loader for ASVspoof2019 LA data (XLS-R + AASIST).
    Only loads audio files since XLS-R processes raw waveforms.
    """
    
    def __init__(self, audio_dir, label_file, max_audio_length=96000):
        self.audio_dir = audio_dir
        self.max_audio_length = max_audio_length
        
        # Load labels and find available audio samples
        self.samples, self.labels = self._load_available_samples(label_file)
        
        print(f"Loaded {len(self.samples)} audio samples")
        if len(self.samples) > 0:
            bonafide_count = sum(1 for label in self.labels if label == 1)
            spoof_count = sum(1 for label in self.labels if label == 0)
            print(f"Label distribution: Spoof={spoof_count}, Bonafide={bonafide_count}")
    
    def _load_available_samples(self, label_file):
        """Load samples that have audio files present."""
        samples = []
        labels = []
        
        # Read label file
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Format: [speaker_id] [filename] [system_id] [attack_type] [label]
                        filename = parts[1]  # Second column is filename
                        label_str = parts[-1]  # Last column is label
                        
                        # Check if audio file exists
                        audio_path = os.path.join(self.audio_dir, f"{filename}.flac")
                        
                        if os.path.exists(audio_path):
                            # Convert label to integer (0 for spoof, 1 for bonafide)
                            label = 1 if label_str.lower() == 'bonafide' else 0
                            samples.append(filename)
                            labels.append(label)
        
        return samples, labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename = self.samples[idx]
        label = self.labels[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, f"{filename}.flac")
        waveform, sr = librosa.load(audio_path, sr=16000)
        
        # Handle audio length
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        elif len(waveform) < self.max_audio_length:
            padding = self.max_audio_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant', constant_values=0)
        
        waveform = torch.from_numpy(waveform).float()
        
        return {
            'waveform': waveform,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': filename
        }


def create_direct_datasets(config, device):
    """
    Create datasets by loading audio data directly from specified paths.
    Uses the existing ASVspoof2019 label files in data/ directory.
    Only requires audio paths since XLS-R + AASIST only uses raw waveforms.
    
    TODO: Fill in the actual audio paths below:
    """
    
    # TODO: FILL IN THESE AUDIO PATHS
    audio_paths = {
        'train': "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_train/flac",  # Path to training audio files (.flac files)
        'dev': "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_dev/flac",    # Path to dev audio files (.flac files)
        'eval': "/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_eval/flac"    # Path to eval audio files (.flac files)
    }
    
    # Label file paths (already exist in data/)
    label_files = {
        'train': 'data/ASVspoof2019.LA.cm.train.trn.txt',
        'dev': 'data/ASVspoof2019.LA.cm.dev.trl.txt', 
        'eval': 'data/ASVspoof2019.LA.cm.eval.trl.txt'
    }
    
    # Check if audio paths are filled
    for split, path in audio_paths.items():
        if not path:
            raise ValueError(f"Please fill in the audio {split} path in audio_paths dictionary")
    
    # Create datasets using our simplified DirectDataset (audio only)
    train_dataset = DirectDataset(
        audio_dir=audio_paths['train'],
        label_file=label_files['train'],
        max_audio_length=config.data.max_audio_length
    )
    
    val_dataset = DirectDataset(
        audio_dir=audio_paths['dev'],
        label_file=label_files['dev'],
        max_audio_length=config.data.max_audio_length
    )
    
    test_dataset = DirectDataset(
        audio_dir=audio_paths['eval'],
        label_file=label_files['eval'],
        max_audio_length=config.data.max_audio_length
    )
    
    return train_dataset, val_dataset, test_dataset


def run_experiment(freeze_backbone, config, device, run_name):
    """Run a single experiment with specified backbone freeze setting."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_name}")
    print(f"Backbone frozen: {freeze_backbone}")
    print(f"{'='*60}")
    
    # Create datasets first to calculate class weights
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_direct_datasets(config, device)
    
    # Calculate current class weights for this experiment
    current_class_weights = config.training.class_weights  # Keep original config unchanged
    if len(train_dataset) > 0:
        # More efficient: directly access labels instead of iterating through dataset
        train_spoof = sum(1 for label in train_dataset.labels if label == 0)
        train_bonafide = sum(1 for label in train_dataset.labels if label == 1)
        
        # Calculate class weights for severe imbalance
        total_samples = len(train_dataset.labels)
        spoof_weight = total_samples / (2 * train_spoof)
        bonafide_weight = total_samples / (2 * train_bonafide)
        calculated_weights = [spoof_weight, bonafide_weight]
        
        # Use calculated weights for this experiment (don't modify original config)
        current_class_weights = calculated_weights
    
    # Initialize wandb for this run (with correct class weights)
    wandb.init(
        project="xlsr-aasist-experiment",
        name=run_name,
        config={
            "model": "XLS-R + AASIST",
            "freeze_backbone": freeze_backbone,
            "fusion_type": "none",
            "data_root": "data",
            "batch_size": config.data.batch_size,
            "class_weights": current_class_weights,
            "device": config.training.device,
            "epochs": config.training.num_epochs if not freeze_backbone else 0,
            "learning_rate": config.training.learning_rate if not freeze_backbone else "N/A"
        }
    )
    
    # Create XLS-R + AASIST model
    print(f"Creating XLS-R + AASIST model (backbone frozen: {freeze_backbone})...")
    model = XLSRAASISTModel(freeze_backbone=freeze_backbone, device=device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True if not freeze_backbone else False, 
        num_workers=config.data.num_workers, 
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Print class distribution for debugging
    if len(train_dataset) > 0:
        train_spoof = sum(1 for label in train_dataset.labels if label == 0)
        train_bonafide = sum(1 for label in train_dataset.labels if label == 1)
        print(f"Train class distribution: Spoof={train_spoof}, Bonafide={train_bonafide}")
        if current_class_weights != config.training.class_weights:
            print(f"Using calculated class weights: [spoof={current_class_weights[0]:.3f}, bonafide={current_class_weights[1]:.3f}]")
    
    if len(val_dataset) > 0:
        val_spoof = sum(1 for label in val_dataset.labels if label == 0)
        val_bonafide = sum(1 for label in val_dataset.labels if label == 1)
        print(f"Val class distribution: Spoof={val_spoof}, Bonafide={val_bonafide}")
    
    if len(test_dataset) > 0:
        test_spoof = sum(1 for label in test_dataset.labels if label == 0)
        test_bonafide = sum(1 for label in test_dataset.labels if label == 1)
        print(f"Test class distribution: Spoof={test_spoof}, Bonafide={test_bonafide}")
    
    # Train model if backbone is not frozen
    if not freeze_backbone:
        # Create a temporary config copy for training
        import copy
        training_config = copy.deepcopy(config)
        training_config.training.class_weights = current_class_weights
        model = train_model(model, train_loader, val_loader, test_loader, device, training_config, run_name)
    
    # Evaluate on all splits (use current_class_weights for consistency)
    print(f"\n📊 EVALUATION RESULTS FOR {run_name}:")
    
    # Train set evaluation
    print("\n📊 TRAIN SET EVALUATION:")
    train_metrics = evaluate_model(model, train_loader, device, current_class_weights)
    print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
    print(f"  Precision: {train_metrics['precision']:.2f}%")
    print(f"  Recall: {train_metrics['recall']:.2f}%")
    print(f"  F1-Score: {train_metrics['f1']:.2f}%")
    print(f"  AUC: {train_metrics['auc']:.2f}%")
    print(f"  EER: {train_metrics['eer']:.2f}%")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    
    # Validation set evaluation
    print("\n📊 VALIDATION SET EVALUATION:")
    val_metrics = evaluate_model(model, val_loader, device, current_class_weights)
    print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"  Precision: {val_metrics['precision']:.2f}%")
    print(f"  Recall: {val_metrics['recall']:.2f}%")
    print(f"  F1-Score: {val_metrics['f1']:.2f}%")
    print(f"  AUC: {val_metrics['auc']:.2f}%")
    print(f"  EER: {val_metrics['eer']:.2f}%")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    
    # Test set evaluation
    print("\n📊 TEST SET EVALUATION:")
    test_metrics = evaluate_model(model, test_loader, device, current_class_weights)
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.2f}%")
    print(f"  Recall: {test_metrics['recall']:.2f}%")
    print(f"  F1-Score: {test_metrics['f1']:.2f}%")
    print(f"  AUC: {test_metrics['auc']:.2f}%")
    print(f"  EER: {test_metrics['eer']:.2f}%")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    
    # Log results to wandb
    wandb.log({
        "train/accuracy": train_metrics['accuracy'],
        "train/precision": train_metrics['precision'],
        "train/recall": train_metrics['recall'],
        "train/f1": train_metrics['f1'],
        "train/auc": train_metrics['auc'],
        "train/eer": train_metrics['eer'],
        "train/loss": train_metrics['loss'],
        
        "val/accuracy": val_metrics['accuracy'],
        "val/precision": val_metrics['precision'],
        "val/recall": val_metrics['recall'],
        "val/f1": val_metrics['f1'],
        "val/auc": val_metrics['auc'],
        "val/eer": val_metrics['eer'],
        "val/loss": val_metrics['loss'],
        
        "test/accuracy": test_metrics['accuracy'],
        "test/precision": test_metrics['precision'],
        "test/recall": test_metrics['recall'],
        "test/f1": test_metrics['f1'],
        "test/auc": test_metrics['auc'],
        "test/eer": test_metrics['eer'],
        "test/loss": test_metrics['loss']
    })
    
    # Log confusion matrices
    for split_name, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        cm = metrics['confusion_matrix']
        # sklearn confusion_matrix orders by label value: [0, 1] = [spoof, bonafide]
        # So cm[0,0] = spoof predicted as spoof, cm[0,1] = spoof predicted as bonafide
        # cm[1,0] = bonafide predicted as spoof, cm[1,1] = bonafide predicted as bonafide
        wandb.log({
            f"{split_name}/confusion_matrix": wandb.Table(
                columns=["", "Predicted Spoofed", "Predicted Bonafide"],
                data=[
                    ["True Spoofed", cm[0, 0], cm[0, 1]],
                    ["True Bonafide", cm[1, 0], cm[1, 1]]
                ]
            )
        })
    
    # Save results to file
    results = {
        "model": "XLS-R + AASIST",
        "run_name": run_name,
        "freeze_backbone": freeze_backbone,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "config": {
            "class_weights": current_class_weights,
            "batch_size": config.data.batch_size,
            "device": config.training.device,
            "epochs": config.training.num_epochs if not freeze_backbone else 0,
            "learning_rate": config.training.learning_rate if not freeze_backbone else "N/A"
        }
    }
    
    # Close wandb
    wandb.finish()
    
    return model, results


def main():
    """Main function for XLS-R + AASIST experiments."""
    
    # Load configuration
    config = default_config
    
    # Setup device
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    # Run 1: Trainable backbone (full training)
    trainable_model, trainable_results = run_experiment(
        freeze_backbone=False, 
        config=config, 
        device=device, 
        run_name="Trainable Backbone"
    )
    
    # Run 2: Frozen backbone (evaluation only)
    frozen_model, frozen_results = run_experiment(
        freeze_backbone=True, 
        config=config, 
        device=device, 
        run_name="Frozen Backbone"
    )
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<20} {'Train EER':<10} {'Val EER':<10} {'Test EER':<10} {'Test F1':<10}")
    print("-" * 70)
    
    print(f"{'Frozen Backbone':<20} {frozen_results['train']['eer']:<10.2f} "
          f"{frozen_results['val']['eer']:<10.2f} {frozen_results['test']['eer']:<10.2f} "
          f"{frozen_results['test']['f1']:<10.2f}")
    
    print(f"{'Trainable Backbone':<20} {trainable_results['train']['eer']:<10.2f} "
          f"{trainable_results['val']['eer']:<10.2f} {trainable_results['test']['eer']:<10.2f} "
          f"{trainable_results['test']['f1']:<10.2f}")
    
    print(f"\n{'='*80}")
    
    return frozen_model, trainable_model, frozen_results, trainable_results


if __name__ == "__main__":
    frozen_model, trainable_model, frozen_results, trainable_results = main() 