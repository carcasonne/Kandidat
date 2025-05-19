import os
from datetime import datetime

import torch
import wandb
from sympy import false
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from tqdm import tqdm
import torch
import timm
from transformers import ASTForAudioClassification
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTForAudioClassification
from torchvision import datasets, transforms

from Datasets import ASVspoofDataset, ADDdataset, FoRdataset, ASVspoofDatasetPretrain, ADDdatasetPretrain, FoRdatasetPretrain
from wandb_login import login
import inspect

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_CHECKPOINT = "/home/alsk/Kandidat/AST/checkpoints/asvspoof-ast-model15_100K_20250506_054106"
ADD_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/ADD"  # Replace with your actual ADD dataset root
FOR_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
ASVS_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms"
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

def get_input_and_labels(is_AST, batch):
    if is_AST:
        inputs = batch["input_values"].to(DEVICE)  # shape: (B, T, 128)
        labels = batch["labels"].to(DEVICE)
        return inputs, labels
    else:
        inputs, labels = batch  # batch is a tuple (inputs, labels)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        return inputs, labels

def load_base_ast_model():
    base_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
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
    return model


# EER
def calculate_eer(y_true, y_score):
    """
    Calculate Equal Error Rate (EER) from the prediction scores.
    
    Args:
        y_true: Ground truth labels (0 for bonafide, 1 for spoof)
        y_score: Model probability outputs for the spoof class
        
    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are equal
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer, thresh


def benchmark_with_probabilities(model, data_loader, flavor_text, is_AST):
    # === Benchmarking Loop ===
    all_probs = []  # Store probability scores
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Benchmarking"):
            inputs, labels = get_input_and_labels(is_AST, batch)

            outputs = model(inputs)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Store probabilities for the "spoof" class (index 1)
            spoof_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(spoof_probs)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for easier processing
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Get binary predictions using 0.5 threshold for standard metrics
    binary_preds = (all_probs > 0.5).astype(int)
    
    # === Calculate EER ===
    eer, threshold = calculate_eer(all_labels, all_probs)
    print(f"\n✅ Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"EER Threshold: {threshold:.4f}")
    
    # === Standard Evaluation Metrics ===
    acc = accuracy_score(all_labels, binary_preds)
    print(f"Benchmark Accuracy: {acc * 100:.2f}%")
    print(classification_report(all_labels, binary_preds, target_names=["bonafide", "spoof"]))

    cm = confusion_matrix(all_labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()

    # === Weights & Biases Logging ===
    wandb.login()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="Benchmark", entity="Holdet_thesis", name=flavor_text + "_" + date)

    wandb.log({
        "EER": eer,
        "EER_Threshold": threshold,
        "Accuracy": acc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn
    })

    # Option to log histograms of probabilities by class
    wandb.log({
        "bonafide_score_distribution": wandb.Histogram(all_probs[all_labels == 0]),
        "spoof_score_distribution": wandb.Histogram(all_probs[all_labels == 1])
    })

    wandb.finish()
    
    return {
        "eer": eer,
        "threshold": threshold,
        "accuracy": acc,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "probabilities": all_probs,
        "labels": all_labels
    }


if __name__ == "__main__":
    print("Starting benchmark with probability output")
    # Load models - this part remains unchanged
    AST_model = load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        finetuned_model_path=AST_MODEL_CHECKPOINT,
        device="cuda"
    )

    # Pretrain_model = load_pretrained_model(saved_model_path=PRETRAIN_MODEL_CHECKPOINT)
    # Pretrain_model.to(DEVICE)

    base_AST_model = load_base_ast_model()
    base_AST_model.to(DEVICE)

    samples = {"bonafide": 100000, "fake":100000} # Load all
    asv_samples = {"bonafide": 10000, "fake": 10000}

    # Dataset loading - unchanged
    # AST Datasets
    #add_test_dataset = ADDdataset(data_dir=ADD_DATASET_PATH, max_per_class=samples)
    #add_test_loader = DataLoader(add_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #for_test_dataset = FoRdataset(data_dir=FOR_DATASET_PATH, max_per_class=samples)
    #for_test_loader = DataLoader(for_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    asvs_test_dataset = ASVspoofDataset(data_dir=ASVS_DATASET_PATH, max_per_class=asv_samples)
    asvs_test_loader = DataLoader(asvs_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Pretrain datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    # pre_add_test_dataset = ADDdatasetPretrain(data_dir=ADD_DATASET_PATH, max_per_class=samples, transform=transform)
    # pre_add_test_loader = DataLoader(pre_add_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # pre_for_test_dataset = FoRdatasetPretrain(data_dir=FOR_DATASET_PATH, max_per_class=samples, transform=transform)
    # pre_for_test_loader = DataLoader(pre_for_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    run_name_1 = "Sanity_check"
    benchmark_with_probabilities(AST_model, asvs_test_loader, run_name_1, True)