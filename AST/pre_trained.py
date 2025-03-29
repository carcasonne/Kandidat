import os
from enum import Enum

import librosa
import numpy as np
from PIL import Image

if __name__ == "__main__":
    import timm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from torch.utils.data import random_split

    class DataType(Enum):
        TRAINING = "training"
        VALIDATION = "validation"
        TESTING = "testing"

    class AudioLabel(Enum):
        FAKE = 0
        REAL = 1

    class AudioData(Dataset):
        def __init__(
                self,
                root_dir: str,
                transform,
                sample_rate: int,
                subset: DataType,
                audio_duration_seconds: int,
        ):
            self.transform = transform
            self.sample_rate = sample_rate
            self.subset = subset
            self.audio_duration_seconds = audio_duration_seconds
            self.files = []

            for label in AudioLabel:
                class_path = os.path.join(root_dir, subset.value, label.name.lower())
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    self.files.append((file_path, label.value))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file_path, label = self.files[idx]
            spec = self.convert_to_spectrogram(file_path)

            # Convert NumPy spectrogram to PIL Image
            spec = (spec - spec.min()) / (spec.max() - spec.min())  # Normalize to 0-1
            spec = (spec * 255).astype(np.uint8)  # Scale to 0-255
            img = Image.fromarray(spec)  # Convert to PIL image

            if self.transform:
                img = self.transform(img)

            return img, label

        def convert_to_spectrogram(self, filepath):
            audio, _ = librosa.load(filepath, sr=self.sample_rate, duration=self.audio_duration_seconds)
            spec = librosa.stft(audio)
            return librosa.amplitude_to_db(np.abs(spec), ref=np.max)


    class ASVSpoofDataset(AudioData):
        def __init__(
                self,
                root_dir: str,  # Root directory of the AVSpoof dataset
                bonafide_keys_path: str,  # Path to the "keys/bonafide" file
                fake_keys_path: str,  # Path to the "keys/fake" file
                transform=None,  # Transform to apply to the audio data
                sample_rate: int = 16000,  # Desired sample rate for audio
                audio_duration_seconds: int = 2,  # Duration of audio clips in seconds
                max_samples_per_class: int = 500  # Limit for samples per class
        ):
            """
            Initializes the dataset, loading file paths and labels from the keys.

            Args:
                root_dir (str): Root directory where the audio files are stored.
                bonafide_keys_path (str): Path to the file containing bonafide sample names.
                fake_keys_path (str): Path to the file containing fake sample names.
                transform (callable, optional): A function/transform to apply to audio data.
                sample_rate (int, optional): Target sample rate for audio data.
                audio_duration_seconds (int, optional): Duration of audio clips to load.
                max_samples_per_class (int, optional): Maximum number of samples per class.
            """
            self.root_dir = root_dir
            self.transform = transform
            self.sample_rate = sample_rate
            self.audio_duration_seconds = audio_duration_seconds
            self.max_samples_per_class = max_samples_per_class

            # Load bonafide and fake sample names
            with open(bonafide_keys_path, 'r') as f:
                bonafide_samples = f.read().strip().split("\n")
            with open(fake_keys_path, 'r') as f:
                fake_samples = f.read().strip().split("\n")

            # Apply the max_samples_per_class limit
            bonafide_samples = bonafide_samples[:max_samples_per_class]
            fake_samples = fake_samples[:max_samples_per_class]

            # Assign labels: 1 for bonafide, 0 for fake
            self.files = [(os.path.join(root_dir, sample + ".flac"), 1) for sample in bonafide_samples] + \
                         [(os.path.join(root_dir, sample + ".flac"), 0) for sample in fake_samples]



    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Pretrained ViT Model
    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True, in_chans=1)

    # Modify classifier head for binary classification
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)  # Binary classification (2 classes)

    # Move model to device
    model.to(device)

    # Define Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 5

    # Define Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    asv_data = ASVSpoofDataset("../ASVspoof2021_DF_eval/flac", "../ASVspoof2021_DF_eval/bonafide", "../ASVspoof2021_DF_eval/fake",
                               transform=transform, sample_rate=16000, audio_duration_seconds=2, max_samples_per_class=500)

    # Define the split ratio
    train_ratio = 0.8
    train_size = int(train_ratio * len(asv_data))  # asv_dataset should be your full dataset
    test_size = len(asv_data) - train_size

    # Split dataset
    train_dataset, test_dataset = random_split(asv_data, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")

    # Validation Loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
