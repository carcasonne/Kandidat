import os
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

login()

wandb.init(project="Kandidat-AST", entity="Holdet_thesis")

# Define dataset path
DATASET_PATH = r"../ASVspoof2021_DF_eval"  # Adjust as needed

# AST Pretrained Model
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Load file names and labels
def load_labels():
    bonafide_path = os.path.join(DATASET_PATH, "bonafide")
    fake_path = os.path.join(DATASET_PATH, "fake")

    labels = []
    
    count = 0
    # Read bonafide labels
    with open(bonafide_path, "r") as f:
        for line in f:
            labels.append((line.strip(), 0))  # Bonafide (Real)

    # Read fake labels
    with open(fake_path, "r") as f:
        for line in f:
            if(count > 5000):
                continue
            count = count + 1
            labels.append((line.strip(), 1))  # Fake (Deepfake)

    # Shuffle the list of labels to randomize the order
    random.shuffle(labels)

    # Convert back to a dictionary if needed (optional)
    labels_dict = {filename: label for filename, label in labels}

    return labels_dict

# Custom Dataset Class
class ASVspoofDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = os.path.join(data_dir, "flac")
        self.feature_extractor = feature_extractor
        self.labels = load_labels()
        
        # Keep only files that exist
        self.files = [f for f in self.labels.keys() if os.path.exists((os.path.join(self.data_dir, f)) + ".flac")]
        self.files = self.files[:50]
        print(f"Found {len(self.files)} valid audio files")  # Debugging line

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = self.labels[file_name]
        file_path = os.path.join(self.data_dir, file_name) + ".flac"

        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if needed
        if sample_rate != self.feature_extractor.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.feature_extractor.sampling_rate)(waveform)

        # Convert to AST-compatible format
        inputs = self.feature_extractor(waveform.numpy(), sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Load dataset
train_dataset = ASVspoofDataset(DATASET_PATH, feature_extractor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load AST Model for Binary Classification
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)

# Modify the classifier for 2 classes
model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)  # Change output layer
model.classifier.out_proj = nn.Linear(2, 2)  # Adjust projection layer

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
num_epochs = 5
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

    # Compute Metrics
    loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    roc_auc = roc_auc_score(true_labels, pred_labels, multi_class='ovr')  # Adjust based on your task

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [acc / 100, precision, recall, f1, roc_auc]  # Normalize accuracy to [0,1]
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
        "ROC AUC": roc_auc,
        "Spider Plot": fig
    })

    if epoch % 10 == 0:
        model.save_pretrained("asvspoof-ast-model")

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}, ROC AUC = {roc_auc:.4f}")



# confuse matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Bonafide", "Fake"], yticklabels=["Bonafide", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()