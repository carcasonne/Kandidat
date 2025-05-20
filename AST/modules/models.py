import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import ASTForAudioClassification

def load_modified_ast_model(base_model_name, finetuned_model_path, device):
    """
    Loads the weights from a tensorflow file into our custom AST architecture
    """

    print(f"Loading base model {base_model_name}")
    # Start with the original pretrained model
    model = ASTForAudioClassification.from_pretrained(base_model_name)

    # Apply architecture modifications
    model.config.max_length = 300
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "spoof"}
    model.config.label2id = {"bonafide": 0, "spoof": 1}

    # Modify the classifier for 2 classes (same as in your training code)
    model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
    model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
    desired_max_length = 350
    position_embeddings = (
        model.audio_spectrogram_transformer.embeddings.position_embeddings
    )
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(
            f"Interpolating position embeddings from {old_len} to {desired_max_length}"
        )
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            size=desired_max_length,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        model.audio_spectrogram_transformer.embeddings.position_embeddings = (
            nn.Parameter(interpolated_pos_emb)
        )

    # Load state dict from the fine-tuned model using safetensors format
    safetensors_path = os.path.join(finetuned_model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        print(f"Loading fine-tuned weights from {finetuned_model_path}")
        finetuned_state_dict = load_file(safetensors_path)
        missing_keys, unexpected_keys = model.load_state_dict(
            finetuned_state_dict, strict=False
        )
        print(
            f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
        )
    else:
        print("Failed to load model from file")
        raise Exception

    print("Model successfully prepared with fine-tuned last layers")

    model = model.to(device)
    return model


def load_pretrained_model(saved_model_path, device):
    """
    Loads the weights from a tensorflow file into the vit_base_patch16_224 architecure
    """
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
    model.to(device)
    return model


def load_base_ast_model(device):
    """
    Loads the weights from a tensorflow file into the MIT/ast-finetuned-audioset-10-10-0.4593 architecure
    """

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
    if hasattr(model.classifier, "dense"):
        model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
        if hasattr(model.classifier, "out_proj"):
            model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
    desired_max_length = 350
    position_embeddings = (
        model.audio_spectrogram_transformer.embeddings.position_embeddings
    )
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(
            f"Interpolating position embeddings from {old_len} to {desired_max_length}"
        )
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            size=desired_max_length,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        model.audio_spectrogram_transformer.embeddings.position_embeddings = (
            nn.Parameter(interpolated_pos_emb)
        )
    model.to(device)
    return model
