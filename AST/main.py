import os
from datetime import datetime

import numpy as np
import librosa
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

from attention_map import calc_attention_maps
from wandb_login import login

login()
wandb.init(project="Kandidat-AST", entity="Holdet_thesis")

samples = {"bonafide": 22600, "fake":300000}
epochs = 20
attention_maps = True


# Define dataset path
DATASET_PATH = r"spectrograms"  # Adjust as needed

# AST Pretrained Model
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)


# Custom Dataset Class
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

        # Ensure correct shape: (1024, 128)
        target_frames = 512
        num_frames, num_mel_bins = spectrogram.shape
        print(f"frames = {num_frames}")

        if num_frames < target_frames:
            # Pad with zeros at the end
            pad_amount = target_frames - num_frames
            spectrogram = np.pad(spectrogram, ((0, pad_amount), (0, 0)), mode='constant')
        elif num_frames > target_frames:
            # Center crop
            start = (num_frames - target_frames) // 2
            spectrogram = spectrogram[start:start + target_frames, :]


        spectrogram = torch.tensor(spectrogram)  # shape: (512, 128)

        return {
            "input_values": spectrogram,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Load dataset
train_dataset = ASVspoofDataset(DATASET_PATH, samples)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load AST Model for Binary Classification
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)

# Modify the classifier for 2 classes
model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)  # Change output layer
model.classifier.out_proj = nn. Linear(2, 2)  # Adjust projection layer

# Freezing to allow inly finetuning some parts
N = 10
for i in range(N):  # Layers 0 to 9
    for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
        param.requires_grad = False

# Update the config
model.config.num_labels = 2
model.config.id2label = {0: "bonafide", 1: "spoof"}
model.config.label2id = {"bonafide": 0, "spoof": 1}

# Move to device
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

true_labels = []
pred_labels = []

# Training Loop
num_epochs = epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    true_labels, pred_labels = [], []  # Lists to store predictions and true labels

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        inputs, labels = batch["input_values"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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
                                                       class_names=["Real","Fake"])})

    # Log to Weights & Biases
    wandb.log({
        "Accuracy": acc,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Spider Plot": fig
    })

    if (epoch % 5 == 0 and epoch > 10) or epoch == 19:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = os.path.join(save_dir,f"asvspoof-ast-model{epoch}_{date}")
        model.save_pretrained(save)

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

#shit is so aids I cant take it anymore
#if attention_maps:
#    calc_attention_maps(model, "AST", device, train_dataset, 20)
