import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from transformers import ASTForAudioClassification

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTForAudioClassification
from Datasets import ASVspoofDataset, ADDdataset

import inspect

# from your_dataset_module import ADDdataset

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "checkpoints/asvspoof-ast-model15_100K_20250506_054106"  # Replace with your actual saved path
ADD_DATASET_PATH = "spectrograms/ADD"  # Replace with your actual ADD dataset root
BATCH_SIZE = 16
NUM_WORKERS = 4

def load_modified_ast_model(model_path):
    """
    Load a previously saved AST model with modified architecture
    for audio spoofing detection.

    Args:
        model_path: Path to the saved model directory

    Returns:
        The loaded model with proper architecture modifications
    """
    print(f"Loading model from {model_path}...")

    # Load the base model from the saved path
    model = ASTForAudioClassification.from_pretrained(model_path)

    # Verify and update configuration if needed
    if model.config.num_labels != 2:
        print(f"Updating num_labels from {model.config.num_labels} to 2")
        model.config.num_labels = 2

    # Set or verify label mappings
    model.config.id2label = {0: "bonafide", 1: "spoof"}
    model.config.label2id = {"bonafide": 0, "spoof": 1}

    # Debug the classifier structure
    print(f"Classifier type: {type(model.classifier).__name__}")

    # Inspect the classifier in more detail
    if hasattr(model.classifier, '__dict__'):
        for attr_name in dir(model.classifier):
            if not attr_name.startswith('_') and not callable(getattr(model.classifier, attr_name)):
                try:
                    attr_value = getattr(model.classifier, attr_name)
                    if isinstance(attr_value, nn.Module):
                        print(f"  - {attr_name}: {type(attr_value).__name__}")
                except:
                    pass

    # Attempt to reconstruct classifier based on the original code
    # We need to adapt to the actual structure of the model
    try:
        if hasattr(model.classifier, 'dense'):
            # Original architecture assumed in your code
            expected_in_features = model.classifier.dense.in_features

            # Modify dense layer
            print(f"Modifying dense layer in_features={expected_in_features}, out_features=2")
            model.classifier.dense = nn.Linear(expected_in_features, 2)

            # Check if out_proj exists and modify if needed
            if hasattr(model.classifier, 'out_proj'):
                print("Modifying out_proj layer out_features=2")
                model.classifier.out_proj = nn.Linear(2, 2)
        else:
            # For ASTMLPHead or other architectures
            print("Alternative classifier structure detected")

            # Try to identify the final layer in the classifier
            final_layer = None
            for name, module in model.classifier.named_modules():
                if isinstance(module, nn.Linear):
                    if hasattr(module, 'out_features'):
                        print(f"Found linear layer: {name} with out_features={module.out_features}")
                        final_layer = (name, module)

            if final_layer and final_layer[1].out_features != 2:
                layer_name, layer = final_layer
                print(f"Modifying {layer_name} to output 2 classes")
                if '.' in layer_name:
                    # Handle nested attributes
                    parts = layer_name.split('.')
                    parent_name = '.'.join(parts[:-1])
                    attr_name = parts[-1]
                    parent = model.classifier
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, attr_name, nn.Linear(layer.in_features, 2))
                else:
                    setattr(model.classifier, layer_name, nn.Linear(layer.in_features, 2))
            elif not final_layer:
                print("WARNING: Could not identify the final classification layer")
    except Exception as e:
        print(f"Error modifying classifier: {e}")
        print("Please inspect the model structure and modify the code accordingly")

    # Check and fix position embeddings if needed
    desired_max_length = 350
    position_embeddings = model.audio_spectrogram_transformer.embeddings.position_embeddings
    current_len = position_embeddings.shape[1]

    if current_len != desired_max_length:
        print(f"Interpolating position embeddings from {current_len} to {desired_max_length}")
        # Use interpolation
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),  # shape: (1, dim, current_len)
            size=desired_max_length,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)  # shape back to (1, new_len, dim)

        model.audio_spectrogram_transformer.embeddings.position_embeddings = nn.Parameter(interpolated_pos_emb)

    # Verify model.config.max_length
    if model.config.max_length != 300:
        print(f"Setting max_length to 300 (was {model.config.max_length})")
        model.config.max_length = 300

    print("Model loaded successfully with proper architecture")
    return model



# === Load the ADD dataset ===
model = load_modified_ast_model("checkpoints/asvspoof-ast-model15_100K_20250506_054106")

samples = {"bonafide": 100000, "fake":100000} # Load all
test_dataset = ADDdataset(data_dir=ADD_DATASET_PATH, max_per_class=samples)
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
