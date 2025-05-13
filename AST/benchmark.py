import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from transformers import ASTForAudioClassification

from AST.main import ADDdataset

# from your_dataset_module import ADDdataset

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "checkpoints/asvspoof-ast-model15_100K_20250506_054106"  # Replace with your actual saved path
ADD_DATASET_PATH = "spectrograms/ADD"  # Replace with your actual ADD dataset root
BATCH_SIZE = 16
NUM_WORKERS = 4

# === Load the model from saved checkpoint ===
model = ASTForAudioClassification.from_pretrained(MODEL_CHECKPOINT)
model.to(DEVICE)
model.eval()

# === Load the ADD dataset ===
test_dataset = ADDdataset(data_dir=ADD_DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# === Benchmarking Loop ===
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Benchmarking on ADD"):
        inputs = batch["input_values"].to(DEVICE)  # shape: (B, T, 128)
        labels = batch["labels"].to(DEVICE)

        outputs = model(inputs)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Evaluation Metrics ===
acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Benchmark Accuracy on ADD: {acc * 100:.2f}%")
print(classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"]))
