import os
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import torch
import timm
from transformers import ASTForAudioClassification

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTForAudioClassification

from AST.Datasets import FoRdataset
from Datasets import ASVspoofDataset, ADDdataset
from wandb_login import login
import inspect

# from your_dataset_module import ADDdataset

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_CHECKPOINT = "checkpoints/asvspoof-ast-model15_100K_20250506_054106"  # Replace with your actual saved path
PRETRAIN_MODEL_CHECKPOINT = "checkpoints/asvspoof-pretrain-model19_20250507_081555"
ADD_DATASET_PATH = "spectrograms/ADD"  # Replace with your actual ADD dataset root
FOR_DATASET_PATH = "spectrograms/FoR/for-2sec/for-2seconds"
BATCH_SIZE = 16


def load_modified_ast_model(base_model_name, finetuned_model_path, device=None):
    """
    Load a model where only the last two layers are replaced with fine-tuned weights.

    Args:
        base_model_name: Name of the original pretrained model to start with
        finetuned_model_path: Path to the saved model with fine-tuned weights
        device: The device to load the model to ('cuda', 'cpu', or None to auto-detect)

    Returns:
        Model with base weights plus fine-tuned last layers
    """
    # Determine device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Loading base model {base_model_name}")
    # Start with the original pretrained model
    model = ASTForAudioClassification.from_pretrained(base_model_name)

    # Apply architecture modifications
    model.config.max_length = 300
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "spoof"}
    model.config.label2id = {"bonafide": 0, "spoof": 1}

    # Modify the classifier for 2 classes (same as in your training code)
    if hasattr(model.classifier, 'dense'):
        model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
        if hasattr(model.classifier, 'out_proj'):
            model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
    desired_max_length = 350
    position_embeddings = model.audio_spectrogram_transformer.embeddings.position_embeddings
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(f"Interpolating position embeddings from {old_len} to {desired_max_length}")
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            size=desired_max_length,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        model.audio_spectrogram_transformer.embeddings.position_embeddings = nn.Parameter(interpolated_pos_emb)

    # Load state dict from the fine-tuned model using safetensors format
    print(f"Loading fine-tuned weights from {finetuned_model_path}")

    try:
        # Try to load using safetensors
        from safetensors import safe_open
        from safetensors.torch import load_file

        safetensors_path = os.path.join(finetuned_model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            print(f"Loading model from safetensors file: {safetensors_path}")
            finetuned_state_dict = load_file(safetensors_path)
        else:
            # Fall back to regular pytorch model loading
            pytorch_path = os.path.join(finetuned_model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_path):
                print(f"Loading model from PyTorch file: {pytorch_path}")
                finetuned_state_dict = torch.load(pytorch_path, map_location=device)
            else:
                # If neither file exists, try loading directly from the path
                print(f"Attempting to load model directly from: {finetuned_model_path}")
                model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)
                # Re-freeze the layers after loading
                N = 10
                for i in range(N):  # Layers 0 to 9
                    for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                        param.requires_grad = False
                model = model.to(device)
                print(f"Model successfully loaded directly")
                return model

    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        print("Trying alternate loading method...")

        # Try loading the model directly using from_pretrained
        try:
            print(f"Loading fine-tuned model directly from {finetuned_model_path}")
            model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)

            # Re-freeze the layers after loading
            N = 10
            for i in range(N):  # Layers 0 to 9
                for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                    param.requires_grad = False

            model = model.to(device)
            print(f"Model successfully loaded directly")
            return model
        except Exception as e2:
            print(f"Failed to load model directly: {e2}")
            raise e2

    # If we got here, we have a state dict to filter
    # Filter the state dict to only include the last two transformer layers and classifier
    last_layers_dict = {}
    for key, value in finetuned_state_dict.items():
        # Include only the last two transformer layers (layers 10 and 11)
        if "encoder.layer.10." in key or "encoder.layer.11." in key:
            last_layers_dict[key] = value
        # Include classifier weights
        elif "classifier" in key:
            last_layers_dict[key] = value

    print(f"Selectively loading {len(last_layers_dict)} weights for the last layers and classifier")

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(last_layers_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # Freeze the first 10 layers to match your training setup
    N = 10
    for i in range(N):  # Layers 0 to 9
        for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
            param.requires_grad = False

    # Move the model to the specified device
    model = model.to(device)
    print(f"Model successfully prepared with fine-tuned last layers")

    return model

def load_pretrained_model(saved_model_path, device=None):
    # Recreate the model architecture
    # Determine device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True, in_chans=1)

    # Modify classifier head for binary classification
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head and last transformer block
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.blocks[-1].parameters():
        param.requires_grad = True

    # Load saved state dict
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    return model


def benchmark(model, data_loader, flavor_text):
    # === Benchmarking Loop ===
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Benchmarking on ADD"):
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

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # === Weights & Biases Logging ===
    wandb.login()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="Benchmark", entity="Holdet_thesis", name=flavor_text + "_" + date)

    wandb.log({
        "Accuracy": acc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn
    })

# === Load the ADD dataset ===
AST_model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=AST_MODEL_CHECKPOINT,      # Your saved model
    device="cuda"
)

Pretrain_model = load_pretrained_model(saved_model_path=PRETRAIN_MODEL_CHECKPOINT)

samples = {"bonafide": 100000, "fake":100000} # Load all

add_test_dataset = ADDdataset(data_dir=ADD_DATASET_PATH, max_per_class=samples)
add_test_loader = DataLoader(add_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for_test_dataset = FoRdataset(data_dir=FOR_DATASET_PATH, max_per_class=samples)
for_test_loader = DataLoader(for_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

g
run_name = f"ASVSpoof_benchmark_ADD"
benchmark(AST_model, add_test_loader, run_name)

run_name1 = f"ASVSpoof_benchmark_FoR"
benchmark(AST_model, for_test_loader, run_name1)

run_name2 = f"Pretrain_benchmark_ADD"
benchmark(AST_model, for_test_loader, run_name2)

run_name3 = f"Pretrain_benchmark_FoR"
benchmark(AST_model, for_test_loader, run_name3)

