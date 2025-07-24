import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random
import librosa
from typing import Dict, List, Tuple, Optional, Union


class FusionAASISTDataset(Dataset):
    """
    Dataset for Fusion + AASIST pipeline using ASVspoof2019 LA dataset.
    
    Features:
    - Reads real labels from ASVspoof2019 label files
    - Handles separate paths for spectrograms, features, and audio
    - Filters out samples missing any modality
    - Uses only complete samples for training/evaluation
    """
    
    def __init__(self, 
                 spectrogram_dir: str,
                 features_dir: str,
                 audio_dir: str,
                 split: str = 'train',
                 sample_rate: int = 16000,
                 max_audio_length: int = 96000,  # 6 seconds at 16kHz
                 use_real_labels: bool = True,
                 seed: int = 42):
        """
        Initialize the dataset.
        
        Args:
            spectrogram_dir: Directory containing spectrogram files (.npy)
            features_dir: Directory containing handcrafted feature files (.npy)
            audio_dir: Directory containing audio files (.flac)
            split: Data split ('train', 'dev', 'test')
            sample_rate: Audio sample rate
            use_real_labels: Whether to use real labels from ASVspoof2019 files
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.spectrogram_dir = spectrogram_dir
        self.features_dir = features_dir
        self.audio_dir = audio_dir
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.use_real_labels = use_real_labels
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Get complete samples (all 3 modalities present)
        self.samples, self.labels = self._get_complete_samples()
        
        print(f"Loaded {len(self.samples)} complete samples from {split} split")
        if len(self.samples) > 0:
            labels_array = np.array(self.labels)
            print(f"Label distribution: {np.bincount(labels_array)}")
            print(f"  Bonafide: {np.sum(labels_array == 1)}")
            print(f"  Spoofed: {np.sum(labels_array == 0)}")
    
    def _get_complete_samples(self) -> Tuple[List[str], List[int]]:
        """
        Get samples that have all 3 modalities (audio, features, spectrograms) present.
        Also reads real labels if use_real_labels is True.
        Only includes samples that are present in the label file.
        """
        if self.use_real_labels:
            # Start with label file to ensure we don't miss any labeled samples
            return self._get_samples_from_labels()
        else:
            # Use dummy labels (only for testing)
            return self._get_samples_from_data()
    
    def _get_samples_from_labels(self) -> Tuple[List[str], List[int]]:
        """
        Start with label file and check which samples have complete data.
        This ensures we don't accidentally exclude labeled samples.
        """
        # Load all samples from label file first
        all_labeled_samples, all_labels = self._load_all_labels()
        
        if not all_labeled_samples:
            print(f"❌ No samples found in label file for {self.split} split!")
            return [], []
        
        print(f"\n🔍 LABEL-BASED SAMPLING - {self.split} split:", flush=True)
        print(f"   - Total samples in label file: {len(all_labeled_samples)}", flush=True)
        
        # Check which samples have complete data
        complete_samples = []
        complete_labels = []
        missing_data_samples = []
        
        for i, sample_id in enumerate(all_labeled_samples):
            has_spectrogram = os.path.exists(os.path.join(self.spectrogram_dir, f"{sample_id}.npy"))
            has_features = os.path.exists(os.path.join(self.features_dir, f"{sample_id}.npy"))
            has_audio = self._find_audio_path(sample_id) is not None
            
            if has_spectrogram and has_features and has_audio:
                complete_samples.append(sample_id)
                complete_labels.append(all_labels[i])
            else:
                missing_data_samples.append(sample_id)
        
        # Report missing data analysis
        print(f"   - Complete samples (all 3 modalities): {len(complete_samples)}", flush=True)
        print(f"   - Missing data samples: {len(missing_data_samples)}", flush=True)
        
        if missing_data_samples:
            print(f"   - Missing data examples: {missing_data_samples[:5]}", flush=True)
            
            # Analyze what's missing
            missing_spectrogram = [s for s in missing_data_samples if not os.path.exists(os.path.join(self.spectrogram_dir, f"{s}.npy"))]
            missing_features = [s for s in missing_data_samples if not os.path.exists(os.path.join(self.features_dir, f"{s}.npy"))]
            missing_audio = [s for s in missing_data_samples if self._find_audio_path(s) is None]
            
            print(f"   - Missing spectrograms: {len(missing_spectrogram)}", flush=True)
            print(f"   - Missing features: {len(missing_features)}", flush=True)
            print(f"   - Missing audio: {len(missing_audio)}", flush=True)
        
        if complete_samples:
            print(f"   - Complete sample examples: {complete_samples[:3]}", flush=True)
            
            # Print label distribution
            bonafide_count = sum(1 for label in complete_labels if label == 1)
            spoof_count = sum(1 for label in complete_labels if label == 0)
            print(f"   - Label distribution: Bonafide={bonafide_count}, Spoof={spoof_count}", flush=True)
        
        if len(complete_samples) == 0:
            print(f"❌ Warning: No complete samples found in {self.split} split!", flush=True)
            return [], []
        
        return complete_samples, complete_labels
    
    def _get_samples_from_data(self) -> Tuple[List[str], List[int]]:
        """
        Original method: start with data files and filter to label file.
        Only used when use_real_labels=False (for testing).
        """
        # Get all available files for each modality
        spectrogram_files = set()
        if os.path.exists(self.spectrogram_dir):
            spectrogram_files = {os.path.splitext(f)[0] for f in os.listdir(self.spectrogram_dir) if f.endswith('.npy')}
        
        features_files = set()
        if os.path.exists(self.features_dir):
            features_files = {os.path.splitext(f)[0] for f in os.listdir(self.features_dir) if f.endswith('.npy')}
        
        audio_files = set()
        if os.path.exists(self.audio_dir):
            for f in os.listdir(self.audio_dir):
                if f.endswith(('.flac', '.wav', '.mp3')):
                    audio_files.add(os.path.splitext(f)[0])
        
        # Find intersection (samples with all 3 modalities)
        complete_samples = spectrogram_files & features_files & audio_files
        
        if len(complete_samples) == 0:
            print(f"❌ Warning: No complete samples found in {self.split} split!", flush=True)
            return [], []
        
        # Sort for reproducibility
        complete_samples = sorted(list(complete_samples))
        
        # Use dummy labels (only for testing)
        labels = [random.randint(0, 1) for _ in range(len(complete_samples))]
        
        return complete_samples, labels
    
    def _load_all_labels(self) -> Tuple[List[str], List[int]]:
        """
        Load all samples and labels from the label file.
        """
        # Map split names to label file names
        split_to_file = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'test': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }
        
        label_file = split_to_file.get(self.split)
        if not label_file:
            print(f"Warning: Unknown split '{self.split}', using dummy labels")
            return [], []
        
        # Look for label file in netscratch location
        label_file_path = os.path.join('/netscratch/eerdogan/fusion_model/data', label_file)
        
        if not os.path.exists(label_file_path):
            print(f"Warning: Label file not found at {label_file_path}")
            print("No samples will be loaded")
            return [], []
        
        # Read label file and create mapping
        sample_ids = []
        labels = []
        
        try:
            with open(label_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            # Format: [ID] [filename] [attack_type] [label]
                            filename = parts[1]  # Second column is filename
                            label_str = parts[-1]  # Last column is label
                            
                            # Convert label to integer (0 for spoof, 1 for bonafide)
                            label = 1 if label_str.lower() == 'bonafide' else 0
                            sample_ids.append(filename)
                            labels.append(label)
        except Exception as e:
            print(f"Error reading label file {label_file_path}: {e}")
            return [], []
        
        return sample_ids, labels
    
    def analyze_missing_data(self) -> Dict:
        """
        Analyze what data is missing and provide detailed statistics.
        """
        # Load all labels first
        all_labeled_samples, all_labels = self._load_all_labels()
        
        if not all_labeled_samples:
            return {
                'total_labeled': 0,
                'complete_samples': 0,
                'missing_data_samples': 0,
                'missing_spectrogram': 0,
                'missing_features': 0,
                'missing_audio': 0,
                'label_distribution': {'bonafide': 0, 'spoof': 0}
            }
        
        # Analyze each sample
        complete_samples = []
        missing_spectrogram_samples = []
        missing_features_samples = []
        missing_audio_samples = []
        
        for i, sample_id in enumerate(all_labeled_samples):
            has_spectrogram = os.path.exists(os.path.join(self.spectrogram_dir, f"{sample_id}.npy"))
            has_features = os.path.exists(os.path.join(self.features_dir, f"{sample_id}.npy"))
            has_audio = self._find_audio_path(sample_id) is not None
            
            if has_spectrogram and has_features and has_audio:
                complete_samples.append(sample_id)
            else:
                if not has_spectrogram:
                    missing_spectrogram_samples.append(sample_id)
                if not has_features:
                    missing_features_samples.append(sample_id)
                if not has_audio:
                    missing_audio_samples.append(sample_id)
        
        # Calculate label distribution for complete samples
        complete_labels = [all_labels[all_labeled_samples.index(s)] for s in complete_samples]
        bonafide_count = sum(1 for label in complete_labels if label == 1)
        spoof_count = sum(1 for label in complete_labels if label == 0)
        
        return {
            'total_labeled': len(all_labeled_samples),
            'complete_samples': len(complete_samples),
            'missing_data_samples': len(all_labeled_samples) - len(complete_samples),
            'missing_spectrogram': len(missing_spectrogram_samples),
            'missing_features': len(missing_features_samples),
            'missing_audio': len(missing_audio_samples),
            'label_distribution': {
                'bonafide': bonafide_count,
                'spoof': spoof_count
            },
            'completeness_percentage': (len(complete_samples) / len(all_labeled_samples)) * 100 if all_labeled_samples else 0
        }
    
    def _load_spectrogram(self, sample_id: str) -> torch.Tensor:
        """Load spectrogram for a sample."""
        file_path = os.path.join(self.spectrogram_dir, f"{sample_id}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spectrogram file not found: {file_path}")
        
        # Load spectrogram
        spectrogram = np.load(file_path)
        
        # Convert to torch tensor
        spectrogram = torch.from_numpy(spectrogram).float()
        
        # Normalize spectrogram (per-channel normalization)
        if spectrogram.dim() == 3:
            # Normalize each channel independently
            for c in range(spectrogram.size(0)):
                channel = spectrogram[c]
                mean = torch.mean(channel)
                std = torch.std(channel)
                if std > 1e-8:  # Avoid division by zero
                    spectrogram[c] = (channel - mean) / std
                else:
                    spectrogram[c] = channel - mean
        
        # Ensure correct shape (should be [4, H, W] for your model)
        if spectrogram.dim() == 2:
            # If it's 2D, add channel dimension
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.dim() == 3 and spectrogram.size(0) != 4:
            # If it's 3D but not 4 channels, we need to handle this
            if spectrogram.size(0) == 1:
                # Repeat single channel to 4 channels
                spectrogram = spectrogram.repeat(4, 1, 1)
            else:
                # Take first 4 channels or pad/truncate
                if spectrogram.size(0) > 4:
                    spectrogram = spectrogram[:4]
                else:
                    # Pad with zeros
                    padding = torch.zeros(4 - spectrogram.size(0), spectrogram.size(1), spectrogram.size(2))
                    spectrogram = torch.cat([spectrogram, padding], dim=0)
        
        return spectrogram
    
    def _load_handcrafted_features(self, sample_id: str) -> torch.Tensor:
        """Load handcrafted features for a sample."""
        file_path = os.path.join(self.features_dir, f"{sample_id}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        # Load features
        features = np.load(file_path)
        
        # Convert to torch tensor
        features = torch.from_numpy(features).float()
        
        # Ensure it's 1D
        if features.dim() > 1:
            features = features.flatten()
        
        # Normalize handcrafted features (z-score normalization)
        mean = torch.mean(features)
        std = torch.std(features)
        if std > 1e-8:  # Avoid division by zero
            features = (features - mean) / std
        else:
            features = features - mean
        
        return features
    
    def _load_waveform(self, sample_id: str) -> torch.Tensor:
        """Load audio waveform for a sample."""
        # Find audio file
        audio_path = self._find_audio_path(sample_id)
        if not audio_path:
            raise FileNotFoundError(f"Audio file not found for sample: {sample_id}")
        
        # Load audio using librosa
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")
        
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).float()
        
        # Normalize waveform (z-score normalization)
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        if std > 1e-8:  # Avoid division by zero
            waveform = (waveform - mean) / std
        else:
            waveform = waveform - mean
        
        # Crop or pad to max_audio_length
        if waveform.size(0) > self.max_audio_length:
            # Crop to max_audio_length (take the first part)
            waveform = waveform[:self.max_audio_length]
        elif waveform.size(0) < self.max_audio_length:
            # Pad with zeros to max_audio_length
            padding = torch.zeros(self.max_audio_length - waveform.size(0))
            waveform = torch.cat([waveform, padding], dim=0)
        
        return waveform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample_id = self.samples[idx]
        label = self.labels[idx]
        
        # Load all modalities
        spectrogram = self._load_spectrogram(sample_id)
        handcrafted_features = self._load_handcrafted_features(sample_id)
        waveform = self._load_waveform(sample_id)
        
        return {
            'spectrogram': spectrogram,
            'handcrafted_features': handcrafted_features,
            'waveform': waveform,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': sample_id
        }
            
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading the data."""
        sample_id = self.samples[idx]
        label = self.labels[idx]
        
        return {
            'sample_id': sample_id,
            'label': label,
            'spectrogram_path': os.path.join(self.spectrogram_dir, f"{sample_id}.npy"),
            'features_path': os.path.join(self.features_dir, f"{sample_id}.npy"),
            'audio_path': self._find_audio_path(sample_id)
        }
    
    def _find_audio_path(self, sample_id: str) -> str:
        """Find the audio file path for a sample."""
        # Try different audio extensions
        for ext in ['.flac', '.wav', '.mp3']:
            audio_path = os.path.join(self.audio_dir, f"{sample_id}{ext}")
            if os.path.exists(audio_path):
                return audio_path
        
        return None


def test_dataset():
    """Test the dataset with the new ASVspoof2019 LA paths."""
    print("Testing FusionAASISTDataset with ASVspoof2019 LA data...")
    
    # Test with train split
    dataset = FusionAASISTDataset(
        spectrogram_dir="/ds-slt/audio/ASVSpoof_LA_19/formant_gabor_feat/gabor_train_npy",
        features_dir="/ds-slt/audio/ASVSpoof_LA_19/opensmile_LA_compare/train",
        audio_dir="/ds-slt/audio/ASVSpoof_LA_19/ASVspoof2019_LA_train/flac",
        split='train',
        max_audio_length=96000,  # 6 seconds
        use_real_labels=True,
        seed=42
    )
    
    if len(dataset) > 0:
        print(f"\nDataset size: {len(dataset)}")
        
        # Test first sample
        sample = dataset[0]
        print(f"\nFirst sample keys: {sample.keys()}")
        print(f"Spectrogram shape: {sample['spectrogram'].shape}")
        print(f"Handcrafted features shape: {sample['handcrafted_features'].shape}")
        print(f"Waveform shape: {sample['waveform'].shape}")
        print(f"Label: {sample['label'].item()} ({'Spoofed' if sample['label'].item() == 0 else 'Bonafide'})")
        print(f"Sample ID: {sample['sample_id']}")
        
        # Test sample info
        info = dataset.get_sample_info(0)
        print(f"\nSample info: {info}")
    else:
        print("No samples found in dataset!")


if __name__ == "__main__":
    test_dataset() 