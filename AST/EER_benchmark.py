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

from Datasets import ASVspoofDataset, ADDdataset, FoRdataset, ASVspoofDatasetPretrain, ADDdatasetPretrain, FoRdatasetPretrain, load_ADD_dataset, load_ASV_dataset, load_FOR_total
from wandb_login import login
import inspect

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_CHECKPOINT = "/home/alsk/Kandidat/AST/checkpoints/asvspoof-ast-model15_100K_20250506_054106"
ADD_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/ADD"  # Replace with your actual ADD dataset root
FOR_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
ASVS_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms"
BATCH_SIZE = 16

def load_modified_ast_model(base_model_name, finetuned_model_path, embedding_size, device=None):
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


    # Interpolate positional embeddings
    desired_max_length = 0
    if embedding_size == 200:
        desired_max_length = 230
    elif embedding_size == 450:
        desired_max_length = 530
    elif embedding_size == 300:
        desired_max_length = 350


    # Apply architecture modifications
    model.config.max_length = embedding_size
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "spoof"}
    model.config.label2id = {"bonafide": 0, "spoof": 1}

    # Modify the classifier for 2 classes (same as in your training code)
    model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
    model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
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

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(finetuned_state_dict, strict=False)
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
    
    embedding_size_asv = 300
    embedding_size_for = 200
    embedding_size_add = 450

    print("Loading datasets")

    # Define sample limits
    samples = {"bonafide": 100000, "fake": 100000}  # Load all
    asv_samples = {"bonafide": 10000, "fake": 10000}

    # Load datasets using the functions from Datasets.py
    # ASVspoof dataset
    asvs_test_loader, _, _ = load_ASV_dataset(
        path=ASVS_DATASET_PATH, 
        samples=asv_samples, 
        is_AST=True, 
        split=None,
        embedding_size=embedding_size_asv
    )
    
    # ADD dataset
    add_test_loader, _, _ = load_ADD_dataset(
        path=ADD_DATASET_PATH, 
        samples=samples, 
        is_AST=True, 
        split=None,
        embedding_size=embedding_size_add
    )
    
    # FOR dataset - using the total dataset function since we want all data
    for_test_loader = load_FOR_total(
        path=FOR_DATASET_PATH, 
        samples=samples, 
        is_AST=True,
        embedding_size=embedding_size_for
    )

    print("Loading models")

    AST_model = load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        finetuned_model_path=AST_MODEL_CHECKPOINT,
        embedding_size=embedding_size_asv,
        device="cuda"
    )
    AST_model.to(DEVICE)

    # Run benchmarks for the AST model only
    run_name_1 = f"AST_benchmark_ASVspoof"
    benchmark_with_probabilities(AST_model, asvs_test_loader, run_name_1, True)

    AST_model = load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        finetuned_model_path=AST_MODEL_CHECKPOINT,
        embedding_size=embedding_size_add,
        device="cuda"
    )
    AST_model.to(DEVICE)

    run_name_2 = f"AST_benchmark_ADD"
    benchmark_with_probabilities(AST_model, add_test_loader, run_name_2, True)

    AST_model = load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        finetuned_model_path=AST_MODEL_CHECKPOINT,
        embedding_size=embedding_size_for,
        device="cuda"
    )
    AST_model.to(DEVICE)

    run_name_3 = f"AST_benchmark_FoR"
    benchmark_with_probabilities(AST_model, for_test_loader, run_name_3, True)