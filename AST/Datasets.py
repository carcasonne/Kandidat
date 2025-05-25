import os
from datetime import datetime
from glob import glob

from torch.utils.data import random_split
import numpy as np
import librosa
import soundfile
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from transformers import AutoFeatureExtractor, ASTForAudioClassification, ASTConfig, ASTModel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import wandb
import plotly.graph_objects as go
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class ASVspoofDataset(Dataset):
    def __init__(self, data_dir, max_per_class=None, transform=None, target_frames=None):
        self.data_dir = data_dir
        self.spec_dir = os.path.join(data_dir, "ASVSpoof")
        self.transform = transform
        self.target_frames = target_frames

        self.class_map = {
            "bonafide": 0,
            "fake": 1
        }

        # If a single int is passed, convert to dict with same value for all classes
        if isinstance(max_per_class, int):
            self.max_per_class = {class_name: max_per_class for class_name in self.class_map}
        else:
            self.max_per_class = max_per_class  # Can be None or a dict per class

        self.files = []
        for class_name, label in self.class_map.items():
            class_folder = os.path.join(self.spec_dir, class_name)
            class_files = [
                os.path.join(class_folder, file)
                for file in os.listdir(class_folder)
                if file.endswith(".npy")
            ]

            max_count = self.max_per_class.get(class_name) if self.max_per_class else None
            if max_count is not None:
                random.shuffle(class_files)
                class_files = class_files[:min(max_count, len(class_files))]

            self.files.extend([(file_path, label) for file_path in class_files])

        print(f"Loaded {len(self.files)} total spectrograms.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T

        # Ensure correct shape: (300, 128)
        # 300 since this is the average
        """"""
        num_frames, num_mel_bins = spectrogram.shape

        if num_frames < self.target_frames:
            # Pad with zeros at the end
            pad_amount = self.target_frames - num_frames
            spectrogram = np.pad(spectrogram, ((0, pad_amount), (0, 0)), mode='constant')
        elif num_frames > self.target_frames:
            # Center crop
            start = (num_frames - self.target_frames) // 2
            spectrogram = spectrogram[start:start + self.target_frames, :]

        spectrogram = torch.tensor(spectrogram)

        if self.transform:
            spectrogram = spectrogram.unsqueeze(0)
            spectrogram = self.transform(spectrogram)
            spectrogram = spectrogram.squeeze(0)

        return {
            "input_values": spectrogram,
            "labels": torch.tensor(label, dtype=torch.long)
        }


class TotalDataset(ASVspoofDataset):
    def __init__(self, root_dir, samples_per_dataset, transform=None, target_frames=None):
        """
        Args:
            root_dir (str): Root directory containing the datasets (e.g., 'spectrograms/')
            samples_per_dataset (dict): Dict specifying number of samples from each dataset
                Example: {'ADD': 100, 'ASVSpoof': 200, 'FoR': 150}
        """
        self.files = []  # List of (filepath, label)
        self.transform = transform
        self.target_frames = target_frames

        for dataset_name, total_samples in samples_per_dataset.items():
            dataset_path = os.path.join(root_dir, dataset_name)
            fake_files = []
            real_files = []

            # Collect files depending on dataset structure
            if dataset_name == 'ADD':
                fake_files = glob(os.path.join(dataset_path, 'fake', '*.npy'))
                real_files = glob(os.path.join(dataset_path, 'genuine', '*.npy'))

            elif dataset_name == 'ASVSpoof':
                fake_files = glob(os.path.join(dataset_path, 'fake', '*.npy'))
                real_files = glob(os.path.join(dataset_path, 'bonafide', '*.npy'))

            elif dataset_name == 'FoR':
                for split in ['Training', 'Testing']:
                    fake_files += glob(os.path.join(dataset_path, 'for-2sec', 'for-2seconds', split, 'Fake', '*.npy'))
                    real_files += glob(os.path.join(dataset_path, 'for-2sec', 'for-2seconds', split, 'Real', '*.npy'))

            # Balance samples
            num_each = total_samples // 2
            sampled_fake = random.sample(fake_files, min(num_each, len(fake_files)))
            sampled_real = random.sample(real_files, min(num_each, len(real_files)))

            self.files.extend([(path, 1) for path in sampled_fake])
            self.files.extend([(path, 0) for path in sampled_real])

        random.shuffle(self.files)

class ADDdataset(ASVspoofDataset):
    def __init__(self, data_dir, max_per_class=None, transform=None, target_frames=None):
        self.data_dir = data_dir  # Root dir containing 'genuine/' and 'fake/'
        self.transform = transform
        self.target_frames = target_frames
        self.class_map = {
            "genuine": 0,
            "fake": 1
        }

        if isinstance(max_per_class, int):
            self.max_per_class = {class_name: max_per_class for class_name in self.class_map}
        else:
            self.max_per_class = max_per_class

        self.files = []
        for class_name, label in self.class_map.items():
            class_folder = os.path.join(self.data_dir, class_name)  # Note: no 'ASVSpoof/' subfolder
            if not os.path.isdir(class_folder):
                raise FileNotFoundError(f"Expected folder '{class_folder}' not found.")

            class_files = [
                os.path.join(class_folder, file)
                for file in os.listdir(class_folder)
                if file.endswith(".npy")
            ]

            max_count = self.max_per_class.get(class_name) if self.max_per_class else None
            if max_count is not None:
                class_files = class_files[:min(max_count, len(class_files))]

            self.files.extend([(file_path, label) for file_path in class_files])

        print(f"[ADDdataset] Loaded {len(self.files)} total spectrograms from '{data_dir}'.")


class FoRdataset(ASVspoofDataset):
    def __init__(self, data_dir, max_per_class=None, transform=None, target_frames=None):
        """
        :param data_dir: Root path to 'FoR/for-2sec/for-2seconds'
        :param max_per_class: Optional int or dict of max samples per class
        :param transform: Optional transform
        """
        self.data_dir = data_dir  # Should be 'FoR/for-2sec/for-2seconds'
        self.transform = transform
        self.target_frames = target_frames
        self.class_map = {
            "Real": 0,
            "Fake": 1
        }

        if isinstance(max_per_class, int):
            self.max_per_class = {class_name: max_per_class for class_name in self.class_map}
        else:
            self.max_per_class = max_per_class  # Can be None or a dict per class

        self.files = []
        for class_name, label in self.class_map.items():
            class_files = []
            for split in ["Training", "Testing"]:
                class_folder = os.path.join(self.data_dir, split, class_name)
                if not os.path.isdir(class_folder):
                    raise FileNotFoundError(f"Expected folder '{class_folder}' not found.")

                split_files = [
                    os.path.join(class_folder, file)
                    for file in os.listdir(class_folder)
                    if file.endswith(".npy")
                ]
                class_files.extend(split_files)

            max_count = self.max_per_class.get(class_name) if self.max_per_class else None
            if max_count is not None:
                random.shuffle(class_files)
                class_files = class_files[:min(max_count, len(class_files))]

            self.files.extend([(file_path, label) for file_path in class_files])

        print(f"[FoRdataset] Loaded {len(self.files)} total spectrograms from '{data_dir}'.")


class FoRdatasetSimple(ASVspoofDataset):
    def __init__(self, data_dir, max_per_class=None, transform=None, target_frames=None):
        """
        :param data_dir: Root path to 'FoR/for-2sec/for-2seconds'
        :param max_per_class: Optional int or dict of max samples per class
        :param transform: Optional transform
        """
        self.data_dir = data_dir  # Should be 'FoR/for-2sec/for-2seconds'
        self.transform = transform
        self.target_frames = target_frames

        self.class_map = {
            "Real": 0,
            "Fake": 1
        }

        if isinstance(max_per_class, int):
            self.max_per_class = {class_name: max_per_class for class_name in self.class_map}
        else:
            self.max_per_class = max_per_class  # Can be None or a dict per class

        self.files = []
        for class_name, label in self.class_map.items():
            class_folder = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_folder):
                raise FileNotFoundError(f"Expected folder '{class_folder}' not found.")

            class_files = [
                os.path.join(class_folder, file)
                for file in os.listdir(class_folder)
                if file.endswith(".npy")
            ]

            max_count = self.max_per_class.get(class_name) if self.max_per_class else None
            if max_count is not None:
                random.shuffle(class_files)
                class_files = class_files[:min(max_count, len(class_files))]

            self.files.extend([(file_path, label) for file_path in class_files])

        print(f"[FoRdataset] Loaded {len(self.files)} total spectrograms from '{data_dir}'.")


class ASVspoofDatasetPretrain(ASVspoofDataset):
    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 128, T)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

class FoRdatasetPretrain(FoRdataset):
    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 128, T)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

class FoRdatasetSimplePretrain(FoRdatasetSimple):
    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 128, T)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

class ADDdatasetPretrain(ADDdataset):
    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 128, T)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

def load_total_dataset(path, samples, split=None, transform=None, embedding_size=None):
    dataset = TotalDataset(path, samples, transform, embedding_size)

    if split is None:
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        return loader, None, None

    # Set validation split ratio
    total_len = len(dataset)
    val_len = int(total_len * split)
    train_len = total_len - val_len

    # Ensure reproducibility
    generator = torch.Generator()
    seed = generator.seed()

    # Split
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=True)
    return train_loader, val_loader, seed

def load_ADD_dataset(path, samples, is_AST, split=None, transform=None, embedding_size=None):
    if is_AST:
        train_dataset = ADDdataset(data_dir=path, max_per_class=samples, target_frames=embedding_size)
    else :
        train_dataset = ADDdatasetPretrain(data_dir=path, max_per_class=samples, transform=transform)

    if split is None:
        loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        return loader, None, None

    # Set validation split ratio
    total_len = len(train_dataset)
    val_len = int(total_len * split)
    train_len = total_len - val_len

    # Ensure reproducibility
    generator = torch.Generator()
    seed = generator.seed()

    # Split
    train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=True)
    return train_loader, val_loader, seed

def load_ASV_dataset(path, samples, is_AST, split=None, transform=None, embedding_size=None):
    if is_AST:
        train_dataset = ASVspoofDataset(data_dir=path, max_per_class=samples, target_frames=embedding_size)
    else :
        train_dataset = ASVspoofDatasetPretrain(data_dir=path, max_per_class=samples, transform=transform)

    if split is None:
        loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        return loader, None, None

    # Set validation split ratio
    total_len = len(train_dataset)
    val_len = int(total_len * split)
    train_len = total_len - val_len

    # Ensure reproducibility
    generator = torch.Generator()
    seed = generator.seed()

    # Split
    train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=True)
    return train_loader, val_loader, seed

def load_FOR_total(path, samples, is_AST, transform=None, embedding_size=None):
    if is_AST:
        dataset = FoRdataset(path, samples, target_frames=embedding_size)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
    else:
        dataset = FoRdatasetPretrain(path, samples, transform=transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader

def load_FOR_dataset(train_path, test_path, is_AST, samples, transform=None, target_size=None):
    if is_AST:
        train_dataset = FoRdatasetSimple(train_path, samples, target_frames=target_size)
        val_dataset = FoRdatasetSimple(test_path, samples, target_frames=target_size)
    else:
        train_dataset = FoRdatasetSimplePretrain(train_path, samples, transform=transform)
        val_dataset = FoRdatasetSimplePretrain(test_path, samples, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    return train_loader, val_loader, None


class StretchMelCropTime:
    def __init__(self, mel_target=224, time_target=224):
        self.mel_target = mel_target
        self.time_target = time_target

    def __call__(self, spectrogram):
        # spectrogram shape: (1, T, 128) - (channels, time, mel_bins)
        C, time_steps, mel_bins = spectrogram.shape

        print(f"Input shape: {spectrogram.shape}")  # Debug print

        # Add batch dimension to make shape (1, 1, T, 128) for F.interpolate
        # F.interpolate expects (N, C, H, W) format
        spectrogram = spectrogram.unsqueeze(0)  # Now: (1, 1, T, 128)

        # Resize: we want to stretch mel_bins (128 -> 224) and keep time_steps
        # F.interpolate size parameter is (H, W) where H=height, W=width
        # In our case: H=time_steps, W=mel_bins
        spectrogram = F.interpolate(
            spectrogram,
            size=(time_steps, self.mel_target),  # (time_steps, mel_target)
            mode='bilinear',
            align_corners=False
        )
        # Now shape: (1, 1, T, 224)

        # Remove batch dimension => shape (1, T, 224)
        spectrogram = spectrogram.squeeze(0)

        print(f"After mel stretching: {spectrogram.shape}")  # Debug print

        # Now crop or pad time dimension to target
        current_time = spectrogram.shape[1]  # T dimension

        if current_time < self.time_target:
            # Pad time dimension
            pad_total = self.time_target - current_time
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # F.pad padding format: (pad_last_dim_left, pad_last_dim_right, pad_second_last_left, pad_second_last_right, ...)
            # For shape (1, T, 224), we want to pad the T dimension (second dimension)
            spectrogram = F.pad(spectrogram, (0, 0, pad_left, pad_right))
        elif current_time > self.time_target:
            # Crop time dimension (center crop)
            start = (current_time - self.time_target) // 2
            spectrogram = spectrogram[:, start:start + self.time_target, :]

        print(f"Final shape: {spectrogram.shape}")  # Debug print

        # Final shape should be (1, 224, 224)
        return spectrogram