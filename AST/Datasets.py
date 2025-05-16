import os
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
import librosa
import soundfile
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, data_dir, max_per_class=None, transform=None):
        self.data_dir = data_dir
        self.spec_dir = os.path.join(data_dir, "ASVSpoof")
        self.transform = transform

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
        target_frames = 300
        num_frames, num_mel_bins = spectrogram.shape

        if num_frames < target_frames:
            # Pad with zeros at the end
            pad_amount = target_frames - num_frames
            spectrogram = np.pad(spectrogram, ((0, pad_amount), (0, 0)), mode='constant')
        elif num_frames > target_frames:
            # Center crop
            start = (num_frames - target_frames) // 2
            spectrogram = spectrogram[start:start + target_frames, :]


        spectrogram = torch.tensor(spectrogram)  # shape: (300, 128)

        return {
            "input_values": spectrogram,
            "labels": torch.tensor(label, dtype=torch.long)
        }

class ADDdataset(ASVspoofDataset):
    def __init__(self, data_dir, max_per_class=None, transform=None):
        self.data_dir = data_dir  # Root dir containing 'genuine/' and 'fake/'
        self.transform = transform

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
    def __init__(self, data_dir, max_per_class=None, transform=None):
        """
        :param data_dir: Root path to 'FoR/for-2sec/for-2seconds'
        :param max_per_class: Optional int or dict of max samples per class
        :param transform: Optional transform
        """
        self.data_dir = data_dir  # Should be 'FoR/for-2sec/for-2seconds'
        self.transform = transform

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
    def __init__(self, data_dir, max_per_class=None, transform=None):
        """
        :param data_dir: Root path to 'FoR/for-2sec/for-2seconds'
        :param max_per_class: Optional int or dict of max samples per class
        :param transform: Optional transform
        """
        self.data_dir = data_dir  # Should be 'FoR/for-2sec/for-2seconds'
        self.transform = transform

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


def load_ADD_dataset(path, samples, is_AST, split=None, transform=None):
    if is_AST:
        train_dataset = ADDdataset(data_dir=path, max_per_class=samples)
    else :
        train_dataset = ADDdatasetPretrain(data_dir=path, max_per_class=samples, transform=transform)

    if split is None:
        loader = DataLoader(train_dataset, batch_size=samples, shuffle=True)
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

def load_ASV_dataset(path, samples, is_AST, split=None, transform=None):
    if is_AST:
        train_dataset = ASVspoofDataset(data_dir=path, max_per_class=samples)
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

def load_FOR_total(path, samples, is_AST, transform=None):
    if is_AST:
        dataset = FoRdataset(path, samples)
        loader = DataLoader(dataset, batch_size=samples, shuffle=True)
    else:
        dataset = FoRdatasetSimplePretrain(path, samples, transform=transform)
        loader = DataLoader(dataset, batch_size=samples, shuffle=True)
    return loader

def load_FOR_dataset(train_path, test_path, is_AST, samples, transform=None):
    if is_AST:
        train_dataset = FoRdatasetSimple(train_path, samples)
        val_dataset = FoRdatasetSimple(test_path, samples)
    else:
        train_dataset = FoRdatasetPretrain(train_path, samples, transform=transform)
        val_dataset = FoRdatasetPretrain(test_path, samples, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    return train_loader, val_loader

class StretchMelCropTime:
    def __init__(self, mel_target=224, time_target=224):
        self.mel_target = mel_target
        self.time_target = time_target

    def __call__(self, spectrogram):
        # spectrogram shape: (1, mel_bins=128, time=X)
        _, mel_bins, time_steps = spectrogram.shape

        # Step 1: Stretch mel bins (frequency axis) from 128 -> 224
        spectrogram = F.interpolate(spectrogram, size=(self.mel_target, time_steps), mode='bilinear', align_corners=False)

        # Step 2: Center crop or pad time axis to 224
        if time_steps < self.time_target:
            pad_total = self.time_target - time_steps
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            spectrogram = F.pad(spectrogram, (pad_left, pad_right))
        elif time_steps > self.time_target:
            start = (time_steps - self.time_target) // 2
            spectrogram = spectrogram[:, :, start:start + self.time_target]

        return spectrogram