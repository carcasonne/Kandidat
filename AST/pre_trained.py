import os
from datetime import datetime
from enum import Enum
import librosa
import numpy as np
import soundfile
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import wandb
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from wandb_login import login

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import random_split

batch_size = 16
learning_rate = 1e-4
num_epochs = 20
pretrain_max_samples = {"bonafide": 2000, "fake":2000}

class DataType(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"

class AudioLabel(Enum):
    FAKE = 0
    REAL = 1


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
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, 128, T)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label



login()

wandb.init(project="Kandidat-Pre-trained", entity="Holdet_thesis")

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ViT Model
model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True, in_chans=1)

# Modify classifier head for binary classification
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, 2)  # Binary classification (2 classes)

for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the final classification layer (classifier head)
for param in model.head.parameters():
    param.requires_grad = True  # Unfreeze the classifier

# Unfreeze the layer before the head (typically the final transformer block)
# In ViT, this is usually the last block in model.blocks
for param in model.blocks[-1].parameters():
    param.requires_grad = True




# Move model to device
model.to(device)

# Define Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

DATASET_PATH = r"spectrograms"
train_dataset = ASVspoofDataset(DATASET_PATH, max_per_class=pretrain_max_samples, transform=transform)

# Set validation split ratio
val_split = 0.2
total_len = len(train_dataset)
val_len = int(total_len * val_split)
train_len = total_len - val_len

# Ensure reproducibility
generator = torch.Generator()
seed = generator.seed()
# Split
train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=generator)

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)



# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    true_labels, pred_labels = [], []  # Lists to store predictions and true labels
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
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Append true labels and predictions to the lists for confusion matrix
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    # Compute Metrics
    loss = running_loss / len(train_loader)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / ((2 * tp) + fp + fn)

    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [acc / 100, precision, recall, f1]  # Normalize accuracy to [0,1]
    values.append(values[0])  # Close the radar chart

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],  # Close the circle
        fill='toself',
        name=f'Epoch {epoch + 1}'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       y_true=true_labels, preds=pred_labels,
                                                       class_names=["Real", "Fake"])})

    # Log to Weights & Biases
    wandb.log({
        "Accuracy": acc,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Spider Plot": fig,
        "Seed": seed
    })

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    val_true_labels, val_pred_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_true_labels.extend(labels.cpu().numpy())
            val_pred_labels.extend(preds.cpu().numpy())

    val_cm = confusion_matrix(val_true_labels, val_pred_labels)
    tn, fp, fn, tp = val_cm.ravel()
    val_loss /= len(val_loader)
    val_acc = (tp + tn) / (tp + tn + fp + fn)
    val_precision = tp / (tp + fp)
    val_recall = tp / (tp + fn)
    val_f1 = (2 * tp) / ((2 * tp) + fp + fn)

    val_values = [val_acc / 100, val_precision, val_recall, val_f1]
    val_values.append(val_values[0])

    val_fig = go.Figure()
    val_fig.add_trace(go.Scatterpolar(
        r=val_values,
        theta=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        fill='toself',
        name=f'Val Epoch {epoch + 1}'
    ))
    val_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

    wandb.log({
        "Val Confusion Matrix": wandb.plot.confusion_matrix(probs=None, y_true=val_true_labels, preds=val_pred_labels,
                                                            class_names=["Real", "Fake"]),
        "Val Accuracy": val_acc,
        "Val Loss": val_loss,
        "Val Precision": val_precision,
        "Val Recall": val_recall,
        "Val F1 Score": val_f1,
        "Val Spider Plot": val_fig
    })
    if (epoch % 5 == 0 and epoch != 0) or epoch == num_epochs - 1:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = os.path.join(save_dir, f"asvspoof-pretrain-model{epoch}_{date}")

        torch.save(model.state_dict(), save)


    print(f"Epoch {epoch + 1}: Train Loss = {loss:.4f}, Train Acc = {acc:.2f}%, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%, "
          f"Val Precision = {val_precision:.4f}, Val Recall = {val_recall:.4f}, Val F1 = {val_f1:.4f}")


"""
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
"""
