import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve, f1_score
import wandb

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/branches'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/fusion'))

from main_fused_assist import FusionAASISTPipeline
from fusion_aasist_dataset import FusionAASISTDataset

# Note: Now using real data from data/train, data/val, data/test directories


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
    """
    Custom collate function to handle variable-length waveforms.
    Pads or truncates waveforms to a fixed length.
    """
    # Separate different types of data
    spectrograms = []
    handcrafted_features = []
    waveforms = []
    labels = []
    sample_ids = []
    
    # Target lengths for padding/truncation
    target_waveform_length = 96000  # 6 seconds at 16kHz
    
    for item in batch:
        spectrograms.append(item['spectrogram'])
        handcrafted_features.append(item['handcrafted_features'])
        
        # Handle variable-length waveforms
        waveform = item['waveform']
        if len(waveform) > target_waveform_length:
            # Truncate
            waveform = waveform[:target_waveform_length]
        elif len(waveform) < target_waveform_length:
            # Pad with zeros
            padding_length = target_waveform_length - len(waveform)
            waveform = torch.cat([waveform, torch.zeros(padding_length)])
        
        waveforms.append(waveform)
        labels.append(item['label'])
        sample_ids.append(item['sample_id'])
    
    # Stack tensors
    return {
        'spectrogram': torch.stack(spectrograms),
        'handcrafted_features': torch.stack(handcrafted_features),
        'waveform': torch.stack(waveforms),
        'label': torch.tensor(labels),
        'sample_id': sample_ids
    }

class FusedAASISTTrainer:
    """
    Trainer for the Fusion + AASIST pipeline
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 device=None,
                 learning_rate=1e-4,
                 weight_decay=1e-3,
                 use_wandb=True,
                 project_name="fusion-aasist",
                 run_name=None,
                 class_weights=None,
                 early_stopping_patience=5,
                 dropout_rate=0.3):
        """
        Initialize the trainer.
        
        Args:
            model: FusionAASISTPipeline model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to run training on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_wandb: Whether to use wandb logging
            project_name: Name of the wandb project
            run_name: Name of the wandb run
            class_weights: Class weights for loss function
            early_stopping_patience: Number of epochs to wait before early stopping
            dropout_rate: Dropout rate for regularization
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Initialize loss function with class weights
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss without class weights")
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_eers = []
        self.train_f1s = []
        self.val_f1s = []
        self.best_val_eer = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract data from batch
            spectrograms = batch['spectrogram'].to(self.device)
            handcrafted_features = batch['handcrafted_features'].to(self.device)
            waveforms = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms, handcrafted_features, waveforms)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with running averages
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='binary') * 100
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_scores = []
        all_predictions = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract data from batch
                spectrograms = batch['spectrogram'].to(self.device)
                handcrafted_features = batch['handcrafted_features'].to(self.device)
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(spectrograms, handcrafted_features, waveforms)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store labels and scores for EER calculation
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(outputs[:, 1].cpu().numpy())  # Use score for class 1 (bonafide)
                all_predictions.extend(predicted.cpu().numpy())
                
                # Update progress bar with running averages
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_accuracy = 100 * correct / total
        
        # Calculate F1 score
        try:
            epoch_f1 = f1_score(all_labels, all_predictions, average='binary') * 100
        except:
            epoch_f1 = 0.0
        
        # Calculate EER
        try:
            eer, eer_threshold = calculate_eer(all_labels, all_scores)
            eer_percentage = eer * 100
        except Exception as e:
            print(f"Warning: Could not calculate EER: {e}")
            eer_percentage = float('inf')
            eer_threshold = 0.0
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)
        self.val_eers.append(eer_percentage)
        self.val_f1s.append(epoch_f1)
        
        return epoch_loss, epoch_accuracy, epoch_f1, eer_percentage, eer_threshold
    
    def test_epoch(self, epoch):
        """Test for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_scores = []
        all_predictions = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc=f'Epoch {epoch+1} - Test')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract data from batch
                spectrograms = batch['spectrogram'].to(self.device)
                handcrafted_features = batch['handcrafted_features'].to(self.device)
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(spectrograms, handcrafted_features, waveforms)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store labels and scores for EER calculation
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(outputs[:, 1].cpu().numpy())  # Use score for class 1 (bonafide)
                all_predictions.extend(predicted.cpu().numpy())
                
                # Update progress bar with running averages
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = total_loss / len(self.test_loader)
        epoch_accuracy = 100 * correct / total
        
        # Calculate F1 score
        try:
            epoch_f1 = f1_score(all_labels, all_predictions, average='binary') * 100
        except:
            epoch_f1 = 0.0
        
        # Calculate EER
        try:
            eer, eer_threshold = calculate_eer(all_labels, all_scores)
            eer_percentage = eer * 100
        except Exception as e:
            print(f"Warning: Could not calculate EER: {e}")
            eer_percentage = float('inf')
            eer_threshold = 0.0
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'f1': epoch_f1,
            'eer': eer_percentage,
            'eer_threshold': eer_threshold
        }
    
    def train(self, num_epochs=5, save_path='checkpoints/fused_assist_model.pth'):
        """Train the model for specified number of epochs."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {self.early_stopping_patience} epochs")
        print(f"Using dropout rate: {self.dropout_rate}")
        
        best_model_path = save_path.replace('.pth', '_best.pth')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_eer, val_eer_threshold = self.validate_epoch(epoch)
            
            # Test phase (if test_loader is provided)
            test_metrics = None
            if self.test_loader is not None:
                test_metrics = self.test_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.train_f1s.append(train_f1)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                self.epochs_without_improvement = 0
                # Save best model
                self.save_model(best_model_path)
                print(f"✅ New best validation EER: {val_eer:.4f}% - Model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"⚠️  No improvement for {self.epochs_without_improvement} epochs")
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs!")
                    print(f"Best validation EER: {self.best_val_eer:.4f}%")
                    break
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'train/f1': train_f1,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/f1': val_f1,
                    'val/eer': val_eer,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                
                if test_metrics:
                    log_dict.update({
                        'test/accuracy': test_metrics['accuracy'],
                        'test/f1': test_metrics['f1'],
                        'test/eer': test_metrics['eer']
                    })
                
                wandb.log(log_dict)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%, EER: {val_eer:.4f}%")
            
            if test_metrics:
                print(f"  Test  - Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.2f}%, EER: {test_metrics['eer']:.4f}%")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                self.save_model(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Load best model for final evaluation
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            print(f"🔄 Loaded best model with validation EER: {self.best_val_eer:.4f}%")
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation EER: {self.best_val_eer:.4f}%")
        
        return {
            'best_val_eer': self.best_val_eer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_eers': self.val_eers
        }
    
    def save_model(self, path):
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_eer': self.best_val_eer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_eers': self.val_eers,
            'train_f1s': self.train_f1s,
            'val_f1s': self.val_f1s
        }, path)
    
    def load_model(self, path):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_eers = checkpoint['val_eers']
        self.best_val_eer = checkpoint['best_val_eer']


def main():
    """Main training function."""
    
    # Load configuration
    from config import default_config
    config = default_config
    
    # Setup
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
    
    # Create real datasets with real labels
    print("Creating datasets from real data with real labels...")
    train_dataset = FusionAASISTDataset(
        data_root='data',
        split='train',
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    val_dataset = FusionAASISTDataset(
        data_root='data',
        split='val',
        use_real_labels=config.data.use_real_labels,
        seed=config.data.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, drop_last=config.data.drop_last)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, drop_last=config.data.drop_last)
    
    # Create trainer
    trainer = FusedAASISTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_wandb=config.training.use_wandb,
        project_name=config.training.project_name,
        run_name=config.training.run_name,
        class_weights=config.training.class_weights
    )

    
    
    # Train for specified epochs
    print(f"Starting training for {config.training.num_epochs} epochs...")
    save_path = os.path.join(config.training.save_dir, f'fused_assist_model_{config.training.run_name}.pth')
    os.makedirs(config.training.save_dir, exist_ok=True)
    history = trainer.train(num_epochs=config.training.num_epochs, save_path=save_path)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {save_path}")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main() 