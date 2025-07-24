#!/usr/bin/env python3
"""
Evaluation script for Fusion + AASIST model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, f1_score
import sys
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import wandb

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/fusion'))

from main_fused_assist import FusionAASISTPipeline
from fusion_aasist_dataset import FusionAASISTDataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def calculate_eer(labels, scores):
    """Calculate Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    frr = 1 - tpr  # False Rejection Rate
    far = fpr      # False Acceptance Rate
    
    # Find threshold where FRR = FAR
    abs_diff = np.abs(frr - far)
    min_idx = np.argmin(abs_diff)
    
    eer = (frr[min_idx] + far[min_idx]) / 2
    threshold = thresholds[min_idx]
    
    return eer, threshold


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length waveforms."""
    spectrograms = [item['spectrogram'] for item in batch]
    handcrafted_features = [item['handcrafted_features'] for item in batch]
    waveforms = [item['waveform'] for item in batch]
    labels = [item['label'] for item in batch]
    
    spectrograms = torch.stack(spectrograms)
    handcrafted_features = torch.stack(handcrafted_features)
    
    target_length = 16000
    padded_waveforms = []
    for waveform in waveforms:
        if waveform.size(0) > target_length:
            padded_waveform = waveform[:target_length]
        else:
            padded_waveform = torch.zeros(target_length)
            padded_waveform[:waveform.size(0)] = waveform
        padded_waveforms.append(padded_waveform)
    
    waveforms = torch.stack(padded_waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'spectrogram': spectrograms,
        'handcrafted_features': handcrafted_features,
        'waveform': waveforms,
        'label': labels
    }


def evaluate_fused_assist_model(model, dataloader, device, class_weights=None):
    """Evaluate Fusion + AASIST model on a dataloader."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_scores = []
    total_loss = 0.0
    
    # Initialize loss function with class weights
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            spectrograms = batch['spectrogram'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            waveforms = batch['waveform'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(spectrograms, handcrafted_features, waveforms)
            loss = criterion(logits, labels)
            
            probabilities = torch.softmax(logits, dim=1)
            _, predictions = logits.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_scores.extend(logits[:, 1].cpu().numpy())  # Use score for class 1 (bonafide)
            total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0
    
    try:
        eer, eer_threshold = calculate_eer(all_labels, all_scores)
    except Exception as e:
        print(f"Warning: Could not calculate EER: {e}")
        eer, eer_threshold = 0.0, 0.0
    
    cm = confusion_matrix(all_labels, all_predictions)
    
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


def main():
    """Main evaluation function."""
    # Load configuration first to get run name
    from config import default_config
    config = default_config
    
    parser = argparse.ArgumentParser(description="Evaluate Fusion + AASIST Model")
    parser.add_argument("--checkpoint", type=str, 
                       default=f"checkpoints/fused_assist_model_{config.training.run_name}.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="data", 
                       help="Path to data directory")
    parser.add_argument("--split", type=str, default="test", 
                       choices=["train", "val", "test"], 
                       help="Data split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Batch size for evaluation (overrides config)")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use wandb logging (overrides config)")
    parser.add_argument("--project_name", type=str, default=None, 
                       help="Name of the wandb project (overrides config)")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Name of the wandb run (overrides config)")
    parser.add_argument("--class_weights", type=float, nargs=2, default=None,
                       help="Class weights [spoof, bonafide] (overrides config)")
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    if args.use_wandb:
        config.evaluation.use_wandb = True
    if args.project_name is not None:
        config.evaluation.project_name = args.project_name
    if args.run_name is not None:
        config.evaluation.run_name = args.run_name
        # Update checkpoint path if run_name is provided
        if args.checkpoint == f"checkpoints/fused_assist_model_{default_config.training.run_name}.pth":
            args.checkpoint = f"checkpoints/fused_assist_model_{args.run_name}.pth"
    if args.class_weights is not None:
        config.training.class_weights = args.class_weights
    
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if config.evaluation.use_wandb:
        wandb.init(
            project=config.evaluation.project_name,
            name=config.evaluation.run_name,
            config={
                "checkpoint": args.checkpoint,
                "data_root": args.data_root,
                "split": args.split,
                "batch_size": config.evaluation.batch_size,
                "model": "FusionAASISTPipeline",
                "fusion_type": config.model.fusion_type
            }
        )
    
    print(f"Loading model from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = FusionAASISTPipeline(
        resnet_ckpt=config.model.resnet_ckpt,
        freeze_backbone=config.model.freeze_backbone,
        freeze_xlsr=config.model.freeze_xlsr,
        fusion_type=config.model.fusion_type,
        fusion_kwargs=config.model.fusion_kwargs,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    print(f"Creating dataset for {args.split} split...")
    dataset = FusionAASISTDataset(
        data_root=args.data_root,
        split=args.split,
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.evaluation.batch_size, 
        shuffle=False, 
        num_workers=config.evaluation.num_workers, 
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    print("Evaluating model...")
    metrics = evaluate_fused_assist_model(model, dataloader, device, class_weights=config.training.class_weights)
    
    # Log to wandb if enabled
    if config.evaluation.use_wandb:
        wandb.log({
            f"{args.split}/loss": metrics['loss'],
            f"{args.split}/accuracy": metrics['accuracy'],
            f"{args.split}/precision": metrics['precision'],
            f"{args.split}/recall": metrics['recall'],
            f"{args.split}/f1": metrics['f1'],
            f"{args.split}/auc": metrics['auc'],
            f"{args.split}/eer": metrics['eer'],
            f"{args.split}/eer_threshold": metrics['eer_threshold']
        })
        
        # Log confusion matrix as a table
        cm = metrics['confusion_matrix']
        # sklearn confusion_matrix orders by label value: [0, 1] = [spoof, bonafide]
        # So cm[0,0] = spoof predicted as spoof, cm[0,1] = spoof predicted as bonafide
        # cm[1,0] = bonafide predicted as spoof, cm[1,1] = bonafide predicted as bonafide
        wandb.log({
            f"{args.split}/confusion_matrix": wandb.Table(
                columns=["Predicted Spoofed", "Predicted Bonafide"],
                data=[
                    ["True Spoofed", cm[0, 0], cm[0, 1]],
                    ["True Bonafide", cm[1, 0], cm[1, 1]]
                ]
            )
        })
    
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {args.split.upper()} SET")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1-Score: {metrics['f1']:.2f}%")
    print(f"AUC: {metrics['auc']:.2f}%")
    print(f"EER: {metrics['eer']:.2f}%")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Spoofed   Bonafide")
    print(f"Actual Spoofed   {metrics['confusion_matrix'][0, 0]:8d}  {metrics['confusion_matrix'][0, 1]:8d}")
    print(f"      Bonafide   {metrics['confusion_matrix'][1, 0]:8d}  {metrics['confusion_matrix'][1, 1]:8d}")
    
    print("="*60)
    
    results_path = f"evaluation_results_{args.split}_{config.evaluation.run_name}.json"
    with open(results_path, 'w') as f:
        save_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items()}
        json.dump(save_metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Close wandb if enabled
    if config.evaluation.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 