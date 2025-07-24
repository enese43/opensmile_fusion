#!/usr/bin/env python3
"""
Main entry point for Fusion + AASIST Audio Spoofing Detection Network
Complete pipeline: Training -> Evaluation
"""

import argparse
import json
import os
import torch
import sys
import wandb

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/fusion'))

from train_fused_assist import FusedAASISTTrainer, custom_collate_fn
from eval_fused_assist import evaluate_fused_assist_model
from main_fused_assist import FusionAASISTPipeline
from fusion_aasist_dataset import FusionAASISTDataset
from torch.utils.data import DataLoader
from config import Config, default_config


def train_and_evaluate(config: Config):
    """Complete pipeline: train Fusion + AASIST model and then evaluate it."""
    
    print("="*60)
    print("STARTING COMPLETE PIPELINE: FUSION + AASIST")
    print("="*60)
    
    # Print configuration
    config.print_config()
    
    # Setup device
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating Fusion + AASIST model...")
    model = FusionAASISTPipeline(
        resnet_ckpt=config.model.resnet_ckpt,
        freeze_backbone=config.model.freeze_backbone,
        freeze_xlsr=config.model.freeze_xlsr,
        fusion_type=config.model.fusion_type,
        fusion_kwargs=config.model.fusion_kwargs,
        device=device
    )
    
    # Create datasets with real labels
    print("Creating datasets from ASVspoof2019 LA data with real labels...")
    train_dataset = FusionAASISTDataset(
        spectrogram_dir=config.data.spectrogram_train,
        features_dir=config.data.features_train,
        audio_dir=config.data.audio_train,
        split='train',
        max_audio_length=config.data.max_audio_length,
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    val_dataset = FusionAASISTDataset(
        spectrogram_dir=config.data.spectrogram_dev,
        features_dir=config.data.features_dev,
        audio_dir=config.data.audio_dev,
        split='dev',
        max_audio_length=config.data.max_audio_length,
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    test_dataset = FusionAASISTDataset(
        spectrogram_dir=config.data.spectrogram_test,
        features_dir=config.data.features_test,
        audio_dir=config.data.audio_test,
        split='test',
        max_audio_length=config.data.max_audio_length,
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    
    # Validate datasets before proceeding
    print("\n" + "="*50, flush=True)
    print("DATASET VALIDATION", flush=True)
    print("="*50, flush=True)
    
    # Check dataset sizes
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    
    print(f"Train dataset size: {train_size}", flush=True)
    print(f"Validation dataset size: {val_size}", flush=True)
    print(f"Test dataset size: {test_size}", flush=True)
    
    # Check if datasets are empty
    if train_size == 0:
        error_msg = "❌ ERROR: Train dataset is empty!"
        print(error_msg, flush=True)
        print("Check the following:", flush=True)
        print(f"  - Spectrogram dir: {config.data.spectrogram_train}", flush=True)
        print(f"  - Features dir: {config.data.features_train}", flush=True)
        print(f"  - Audio dir: {config.data.audio_train}", flush=True)
        print("  - Label file exists and is readable", flush=True)
        return None, None, None
    
    if val_size == 0:
        error_msg = "❌ ERROR: Validation dataset is empty!"
        print(error_msg, flush=True)
        print("Check the following:", flush=True)
        print(f"  - Spectrogram dir: {config.data.spectrogram_dev}", flush=True)
        print(f"  - Features dir: {config.data.features_dev}", flush=True)
        print(f"  - Audio dir: {config.data.audio_dev}", flush=True)
        print("  - Label file exists and is readable", flush=True)
        return None, None, None
    
    if test_size == 0:
        warning_msg = "❌ ERROR: Test dataset is empty!"
        print(warning_msg, flush=True)
        print("Check the following:", flush=True)
        print(f"  - Spectrogram dir: {config.data.spectrogram_test}", flush=True)
        print(f"  - Features dir: {config.data.features_test}", flush=True)
        print(f"  - Audio dir: {config.data.audio_test}", flush=True)
        print("  - Label file exists and is readable", flush=True)
        print("⚠️  WARNING: Training will proceed but test evaluation will be skipped!", flush=True)
    else:
        success_msg = "✅ All datasets loaded successfully!"
        print(success_msg, flush=True)
    
    # Test loading a sample from each dataset
    print("\nTesting data loading...", flush=True)
    try:
        # Test train sample
        train_sample = train_dataset[0]
        train_sample_id = train_sample['sample_id']
        print(f"✅ Train sample loaded: {train_sample_id}", flush=True)
        
        # Test val sample
        val_sample = val_dataset[0]
        val_sample_id = val_sample['sample_id']
        print(f"✅ Validation sample loaded: {val_sample_id}", flush=True)
        
        # Test test sample if available
        test_sample_id = None
        if test_size > 0:
            test_sample = test_dataset[0]
            test_sample_id = test_sample['sample_id']
            print(f"✅ Test sample loaded: {test_sample_id}", flush=True)
        
        success_msg = "✅ Data loading test passed!"
        print(success_msg, flush=True)
        
    except Exception as e:
        error_msg = f"❌ ERROR: Failed to load sample data: {e}"
        print(error_msg, flush=True)
        print("Check if all data files are accessible and in correct format", flush=True)
        return None, None, None
    
    print("✅ Data loading test passed!")
    print("="*50, flush=True)
    
    # Analyze missing data for each split
    print("\n" + "="*50, flush=True)
    print("DATASET COMPLETENESS ANALYSIS", flush=True)
    print("="*50, flush=True)
    
    for split_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        print(f"\n📊 {split_name.upper()} DATASET ANALYSIS:", flush=True)
        analysis = dataset.analyze_missing_data()
        
        print(f"   - Total samples in label file: {analysis['total_labeled']}", flush=True)
        print(f"   - Complete samples (all 3 modalities): {analysis['complete_samples']}", flush=True)
        print(f"   - Missing data samples: {analysis['missing_data_samples']}", flush=True)
        print(f"   - Completeness: {analysis['completeness_percentage']:.1f}%", flush=True)
        
        if analysis['missing_data_samples'] > 0:
            print(f"   - Missing spectrograms: {analysis['missing_spectrogram']}", flush=True)
            print(f"   - Missing features: {analysis['missing_features']}", flush=True)
            print(f"   - Missing audio: {analysis['missing_audio']}", flush=True)
        
        if analysis['complete_samples'] > 0:
            print(f"   - Label distribution: Bonafide={analysis['label_distribution']['bonafide']}, Spoof={analysis['label_distribution']['spoof']}", flush=True)
        
        if analysis['completeness_percentage'] < 90:
            print(f"   ⚠️  WARNING: Low completeness ({analysis['completeness_percentage']:.1f}%) - consider investigating missing data", flush=True)
    
    print("="*50, flush=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        collate_fn=custom_collate_fn, 
        drop_last=config.data.drop_last
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        collate_fn=custom_collate_fn, 
        drop_last=config.data.drop_last
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.evaluation.batch_size, 
        shuffle=False, 
        num_workers=config.evaluation.num_workers, 
        collate_fn=custom_collate_fn, 
        drop_last=False
    )
    
    # PHASE 1: TRAINING
    print("\n" + "="*40)
    print("PHASE 1: TRAINING")
    print("="*40)
    
    # Initialize wandb if requested
    if config.training.use_wandb:
        print(f"Initializing wandb with project: {config.training.project_name}, run: {config.training.run_name}")
        wandb.init(
            project=config.training.project_name,
            name=config.training.run_name,
            config={
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "class_weights": config.training.class_weights,
                "model": "FusionAASISTPipeline",
                "fusion_type": config.model.fusion_type,
                "freeze_backbone": config.model.freeze_backbone,
                "freeze_xlsr": config.model.freeze_xlsr,
                "batch_size": config.data.batch_size,
                "num_epochs": config.training.num_epochs,
                "dataset/train_size": len(train_dataset),
                "dataset/val_size": len(val_dataset),
                "dataset/test_size": len(test_dataset),
                "dataset/total_samples": len(train_dataset) + len(val_dataset) + len(test_dataset)
            }
        )
        print("✅ Wandb initialized successfully!")
    
    # Create trainer
    trainer = FusedAASISTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_wandb=config.training.use_wandb,
        project_name=config.training.project_name,
        run_name=config.training.run_name,
        class_weights=config.training.class_weights
    )
    
    print(f"\n✅ Trainer created successfully!", flush=True)
    
    # Verify test dataset is loaded correctly
    print(f"\n" + "="*50, flush=True)
    print("TEST DATASET VERIFICATION", flush=True)
    print("="*50, flush=True)
    
    if len(test_dataset) > 0:
        print(f"✅ Test dataset loaded successfully!", flush=True)
        print(f"   - Test dataset size: {len(test_dataset)}", flush=True)
        print(f"   - Test dataloader batches: {len(test_loader)}", flush=True)
        
        # Test loading a sample from test dataset
        try:
            test_sample = test_dataset[0]
            test_sample_id = test_sample['sample_id']
            print(f"   - Test sample loaded: {test_sample_id}", flush=True)
            print(f"   - Test sample shape - spectrogram: {test_sample['spectrogram'].shape}", flush=True)
            print(f"   - Test sample shape - features: {test_sample['handcrafted_features'].shape}", flush=True)
            print(f"   - Test sample shape - waveform: {test_sample['waveform'].shape}", flush=True)
            print(f"   - Test sample label: {test_sample['label']}", flush=True)
            print(f"✅ Test dataset verification passed!", flush=True)
        except Exception as e:
            print(f"❌ ERROR: Failed to load test sample: {e}", flush=True)
            return None, None, None
    else:
        print(f"❌ WARNING: Test dataset is empty!", flush=True)
        print(f"   - Test dataset size: {len(test_dataset)}", flush=True)
        print(f"   - Test evaluation will be skipped!", flush=True)
        
        # Show detailed path information for debugging
        print(f"\n🔍 DEBUGGING TEST DATASET PATHS:", flush=True)
        print(f"   - Spectrogram dir: {config.data.spectrogram_test}", flush=True)
        print(f"   - Features dir: {config.data.features_test}", flush=True)
        print(f"   - Audio dir: {config.data.audio_test}", flush=True)
        
        # Check if directories exist
        print(f"\n📁 DIRECTORY EXISTENCE CHECK:", flush=True)
        print(f"   - Spectrogram dir exists: {os.path.exists(config.data.spectrogram_test)}", flush=True)
        print(f"   - Features dir exists: {os.path.exists(config.data.features_test)}", flush=True)
        print(f"   - Audio dir exists: {os.path.exists(config.data.audio_test)}", flush=True)
        
        # List contents of directories if they exist
        print(f"\n📂 DIRECTORY CONTENTS:", flush=True)
        if os.path.exists(config.data.spectrogram_test):
            try:
                spectro_files = os.listdir(config.data.spectrogram_test)
                print(f"   - Spectrogram files ({len(spectro_files)}): {spectro_files[:5]}{'...' if len(spectro_files) > 5 else ''}", flush=True)
            except Exception as e:
                print(f"   - Error listing spectrogram dir: {e}", flush=True)
        else:
            print(f"   - Spectrogram directory does not exist", flush=True)
            
        if os.path.exists(config.data.features_test):
            try:
                feature_files = os.listdir(config.data.features_test)
                print(f"   - Feature files ({len(feature_files)}): {feature_files[:5]}{'...' if len(feature_files) > 5 else ''}", flush=True)
            except Exception as e:
                print(f"   - Error listing features dir: {e}", flush=True)
        else:
            print(f"   - Features directory does not exist", flush=True)
            
        if os.path.exists(config.data.audio_test):
            try:
                audio_files = os.listdir(config.data.audio_test)
                print(f"   - Audio files ({len(audio_files)}): {audio_files[:5]}{'...' if len(audio_files) > 5 else ''}", flush=True)
            except Exception as e:
                print(f"   - Error listing audio dir: {e}", flush=True)
        else:
            print(f"   - Audio directory does not exist", flush=True)
        
        # Check label file
        split_to_file = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'test': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }
        label_file = split_to_file.get('test')
        label_file_path = os.path.join('/netscratch/eerdogan/fusion_model/data', label_file)
        print(f"\n📄 LABEL FILE CHECK:", flush=True)
        print(f"   - Label file path: {label_file_path}", flush=True)
        print(f"   - Label file exists: {os.path.exists(label_file_path)}", flush=True)
        if os.path.exists(label_file_path):
            try:
                with open(label_file_path, 'r') as f:
                    label_lines = f.readlines()
                print(f"   - Label file lines: {len(label_lines)}", flush=True)
                if label_lines:
                    print(f"   - First few labels: {label_lines[:3]}", flush=True)
            except Exception as e:
                print(f"   - Error reading label file: {e}", flush=True)
    
    print("="*50, flush=True)
    
    # Train the model
    print("Starting training...")
    save_path = os.path.join(config.training.save_dir, f'fused_assist_model_{config.training.run_name}.pth')
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    history = trainer.train(num_epochs=config.training.num_epochs, save_path=save_path)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {save_path}")
    
    # PHASE 2: EVALUATION
    print("\n" + "="*40)
    print("PHASE 2: EVALUATION")
    print("="*40)
    
    # Check if test dataset is available for evaluation
    if len(test_dataset) == 0:
        print("❌ Test dataset is empty! Skipping evaluation.")
        print("Training completed successfully, but no test evaluation possible.")
        return trainer, history, None
    
    # Load the best model for evaluation
    print("Loading best model for evaluation...")
    if not os.path.exists(save_path):
        print(f"Error: Model checkpoint not found at {save_path}")
        return trainer, history, None
    
    trainer.load_model(save_path)
    print(f"Loaded model with best validation EER: {trainer.best_val_eer:.2f}%")
    
    # Verify model is loaded correctly
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test dataloader batches: {len(test_loader)}")
    
    # Evaluate on test set
    print("Evaluating best model on test set...")
    test_metrics = evaluate_fused_assist_model(model, test_loader, device, class_weights=config.training.class_weights)
    
    if test_metrics is None:
        print("Error: Test evaluation failed!")
        return trainer, history, None
    
    # Print final results to terminal
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {test_metrics['precision']:.2f}%")
    print(f"Test Recall: {test_metrics['recall']:.2f}%")
    print(f"Test F1-Score: {test_metrics['f1']:.2f}%")
    print(f"Test AUC: {test_metrics['auc']:.2f}%")
    print(f"Test EER: {test_metrics['eer']:.2f}%")
    print(f"Test EER Threshold: {test_metrics['eer_threshold']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    print(f"\nTest Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Spoofed  Bonafide")
    print(f"Actual Spoofed   {test_metrics['confusion_matrix'][0, 0]:8d}  {test_metrics['confusion_matrix'][0, 1]:8d}")
    print(f"Actual Bonafide  {test_metrics['confusion_matrix'][1, 0]:8d}  {test_metrics['confusion_matrix'][1, 1]:8d}")
    print("="*60)
    
    # Log test results to wandb if enabled
    if config.training.use_wandb:
        print("Logging test results to wandb...")
        try:
            wandb.log({
                "test/loss": test_metrics['loss'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall'],
                "test/f1": test_metrics['f1'],
                "test/auc": test_metrics['auc'],
                "test/eer": test_metrics['eer'],
                "test/eer_threshold": test_metrics['eer_threshold'],
                "best_val_eer": trainer.best_val_eer
            })
            
            # Log confusion matrix as a table
            cm = test_metrics['confusion_matrix']
            wandb.log({
                "test/confusion_matrix": wandb.Table(
                    columns=["Actual", "Predicted Spoofed", "Predicted Bonafide"],
                    data=[
                        ["Spoofed", cm[0, 0], cm[0, 1]],
                        ["Bonafide", cm[1, 0], cm[1, 1]]
                    ]
                )
            })
            print("Test results logged to wandb successfully!")
        except Exception as e:
            print(f"Warning: Failed to log test results to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to file
    results_path = f'final_results_{config.training.run_name}.json'
    with open(results_path, 'w') as f:
        save_metrics = {k: v.tolist() if hasattr(v, 'tolist') else v 
                       for k, v in test_metrics.items()}
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary of key metrics
    print("\n" + "="*40)
    print("SUMMARY OF KEY METRICS")
    print("="*40)
    print(f"Best Validation EER: {trainer.best_val_eer:.2f}%")
    print(f"Test EER: {test_metrics['eer']:.2f}%")
    print(f"Test F1-Score: {test_metrics['f1']:.2f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print("="*40)
    
    # Close wandb if enabled
    if config.training.use_wandb:
        wandb.finish()
    
    return trainer, history, test_metrics


def main():
    """Main function with command-line interface."""
    print("Using default configuration")
    config = default_config
    
    # Run complete pipeline
    train_and_evaluate(config)


if __name__ == "__main__":
    main() 