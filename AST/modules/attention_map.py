import os
from collections import defaultdict

import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import cv2
from torchvision import datasets, transforms

from modules.models import load_pretrained_model_attention
from Datasets import *
from modules.models import load_modified_ast_model, load_pretrained_model

from Datasets import ASVspoofDataset


def generate_attention_maps(model, inputs, labels, preds, device, output_dir, num_samples=5):
    """
    Generate attention maps for correctly and incorrectly classified samples.

    Args:
        model: Transformer-based model with attention.
        inputs: Tensor of input samples (batch form).
        labels: Ground-truth labels.
        preds: Model predictions.
        device: torch device.
        output_dir: Path to save attention maps.
        num_samples: Number of correct and wrong samples to visualize.
    """
    model.eval()

    # Move everything to CPU or device
    inputs = inputs.to(device)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    correct_indices = [i for i in range(len(labels)) if preds[i] == labels[i]]
    wrong_indices = [i for i in range(len(labels)) if preds[i] != labels[i]]

    os.makedirs(output_dir, exist_ok=True)

    def plot_attention(input_tensor, attention, idx, label, pred, tag):
        fig, axs = plt.subplots(1, len(attention), figsize=(15, 3))
        for i, attn in enumerate(attention):
            axs[i].imshow(attn[0].cpu().detach(), cmap='viridis')
            axs[i].set_title(f"Layer {i}")
            axs[i].axis('off')
        fig.suptitle(f"{tag} Sample {idx} - Label: {label}, Pred: {pred}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tag}_sample_{idx}_label_{label}_pred_{pred}.png"))
        plt.close()

    def extract_attention(input_tensor):
        # Forward pass with hooks if model supports attention output
        output = model(input_tensor.unsqueeze(0))
        if hasattr(output, "attentions"):
            return output.attentions
        elif hasattr(model, "get_attentions"):  # Optional: a custom method
            return model.get_attentions(input_tensor.unsqueeze(0))
        else:
            raise ValueError("Model does not provide attention outputs.")

    for tag, indices in [("correct", correct_indices), ("wrong", wrong_indices)]:
        for i in indices[:num_samples]:
            input_tensor = inputs[i]
            attention = extract_attention(input_tensor)
            plot_attention(input_tensor, attention, i, labels[i], preds[i], tag)


def generate_enhanced_attention_maps_pretrain(model, dataset, num_samples=5, flavor_text="", save_dir="attention-maps-enhanced",
                                     device="cuda"):
    """
    Generate visualizations with three panels:
    1. Original spectrogram
    2. Attention map alone
    3. Overlaid visualization
    """
    import os
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from collections import defaultdict

    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()

    # Ensure all classes are considered before stopping
    class_samples = defaultdict(list)
    for sample in dataset.dataset:
        spectrogram, label = sample  # unpack the tuple
        if len(class_samples[label]) < num_samples:
            class_samples[label].append((spectrogram, label))  # store as tuple again

    selected_samples = class_samples[0] + class_samples[1]

    with torch.no_grad():
        for i, sample in enumerate(tqdm(selected_samples, desc="Generating attention maps")):
            spectrogram, label = sample
            input_tensor = spectrogram.unsqueeze(0).to(device)

            # Forward pass with attention
            outputs = model(input_tensor)
            preds = torch.argmax(outputs, dim=1)
            pred_label = preds.item()

            # Get last layer attention
            attn = model.last_attn_map[0]  # Shape: (num_heads, num_tokens, num_tokens)

            # Average over all heads
            attn_avg = attn.mean(dim=0)  # shape: (num_tokens, num_tokens)

            # For classification tasks: Get CLS token attention to all other tokens
            cls_attn = attn_avg[0, 1:]  # CLS token's attention to other tokens, shape: (num_patches,)

            # Get original spectrogram dimensions
            spectrogram_np = spectrogram.squeeze().cpu().numpy()

            # Handle potential dimension variations
            if len(spectrogram_np.shape) == 3:
                spectrogram_np = spectrogram_np[0]  # Take first channel if multi-channel

            original_height, original_width = spectrogram_np.shape


            # Calculate patch dimensions
            # Standard ViT uses 16x16 patches, but let's verify from the attention map size
            patch_size = 16

            # The ViT model might internally resize the input, so we need to work backwards
            # from the number of attention tokens to determine the grid size
            num_patches = len(cls_attn)

            # Find the closest square or rectangular grid that matches num_patches
            # and is reasonable for the original image dimensions
            possible_heights = []
            possible_widths = []

            for h in range(1, int(np.sqrt(num_patches)) + 1):
                if num_patches % h == 0:
                    w = num_patches // h
                    possible_heights.append(h)
                    possible_widths.append(w)

            # Choose the grid dimensions that best match the aspect ratio of the original image
            original_aspect_ratio = original_width / original_height
            best_idx = 0
            best_ratio_diff = float('inf')

            for idx, (h, w) in enumerate(zip(possible_heights, possible_widths)):
                grid_aspect_ratio = w / h
                ratio_diff = abs(grid_aspect_ratio - original_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_idx = idx

            grid_height = possible_heights[best_idx]
            grid_width = possible_widths[best_idx]


            # Ensure we don't exceed available tokens
            tokens_needed = grid_height * grid_width
            if tokens_needed > len(cls_attn):
                print(f"Warning: Need {tokens_needed} tokens but only have {len(cls_attn)}")
                # Fallback to square grid
                grid_size = int(np.sqrt(len(cls_attn)))
                grid_height = grid_width = grid_size
                tokens_needed = grid_size * grid_size

            # Reshape attention to 2D grid
            attn_tokens_used = cls_attn[:tokens_needed].cpu().numpy()
            attn_2d = attn_tokens_used.reshape(grid_height, grid_width)


            # Resize attention map to match original spectrogram dimensions
            # cv2.resize expects (width, height) but returns (height, width)
            attn_resized = cv2.resize(attn_2d, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


            # Normalize attention map
            attn_normalized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            spectrogram_np = spectrogram_np.T
            attn_normalized = attn_normalized.T
            # Plot 1: Original spectrogram
            im1 = axes[0].imshow(spectrogram_np, origin='lower', aspect='auto', cmap='viridis')
            axes[0].set_title("Original Spectrogram")
            axes[0].axis("off")
            fig.colorbar(im1, ax=axes[0], shrink=0.8)

            # Plot 2: Attention map alone
            im2 = axes[1].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet')
            axes[1].set_title("Attention Map")
            axes[1].axis("off")
            fig.colorbar(im2, ax=axes[1], shrink=0.8)

            # Plot 3: Overlay
            im3 = axes[2].imshow(spectrogram_np, origin='lower', aspect='auto', cmap='viridis')
            im4 = axes[2].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet', alpha=0.4)
            axes[2].set_title("Attention Overlay")
            axes[2].axis("off")

            fig.suptitle(
                f"{flavor_text} Sample {i} - Label: {'bonafide' if label == 0 else 'deepfake'} - Prediction: {'bonafide' if pred_label == 0 else 'deepfake'}",
                fontsize=16
            )

            # Save the figure
            save_path = os.path.join(save_dir, f"attention_visualization_{flavor_text}_{i}_label_{label}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()

            print(f"Saved visualization to: {save_path}")
            print("-" * 50)

def generate_enhanced_attention_maps(model, dataset, num_samples=5, flavor_text="", save_dir="attention-maps-enhanced", device="cuda"):
    """
    Generate visualizations with three panels:
    1. Original spectrogram
    2. Attention map alone
    3. Overlaid visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()

    # Ensure all classes are considered before stopping
    class_samples = defaultdict(list)
    for sample in dataset.dataset:
        label = sample["labels"].item()
        if len(class_samples[label]) < num_samples:
            class_samples[label].append(sample)

    selected_samples = class_samples[0] + class_samples[1]

    with torch.no_grad():
        for i, sample in enumerate(tqdm(selected_samples, desc="Generating attention maps")):
            input_tensor = sample["input_values"].unsqueeze(0).to(device)
            label = sample["labels"].item()

            # Forward pass with attention
            outputs = model(
                input_values=input_tensor,
                output_attentions=True,
                return_dict=True
            )
            preds = torch.argmax(outputs.logits, dim=1)
            pred_label = preds.item()

            # Get last layer attention
            attn = outputs.attentions[-1]  # shape: (1, num_heads, seq_len, seq_len)
            attn = attn[0]  # (num_heads, seq_len, seq_len)

            # Average over all heads
            attn_avg = attn.mean(dim=0)  # shape: (seq_len, seq_len)

            # For classification tasks: Get CLS token attention to all other tokens
            cls_attn = attn_avg[0, 1:]  # CLS token's attention to other tokens
            attn_map_1d = cls_attn.cpu().numpy()

            # Use the square reshape approach
            seq_len = attn_map_1d.shape[0]
            sqrt_len = int(np.ceil(np.sqrt(seq_len)))
            pad_len = sqrt_len ** 2 - seq_len
            attn_padded = np.pad(attn_map_1d, (0, pad_len), mode="constant")
            attn_square = attn_padded.reshape(sqrt_len, sqrt_len)

            # Get spectrogram dimensions
            spec_height, spec_width = input_tensor.shape[1:]  # Should be (300, 128)

            # Resize attention map to match spectrogram dimensions
            attn_resized = cv2.resize(attn_square, (spec_width, spec_height))

            # Load original spectrogram
            spectrogram = sample["input_values"].cpu().numpy()  # shape: (300, 128)

            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            spectrogram = spectrogram.T

            # Plot 1: Original spectrogram
            im1 = axes[0].imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            axes[0].set_title("Original Spectrogram")
            axes[0].axis("off")
            fig.colorbar(im1, ax=axes[0], shrink=0.8)

            # Plot 2: Attention map alone
            # Normalize attention for better visualization
            attn_normalized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            attn_normalized = attn_normalized.T

            im2 = axes[1].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet')
            axes[1].set_title("Attention Map")
            axes[1].axis("off")
            fig.colorbar(im2, ax=axes[1], shrink=0.8)

            # Plot 3: Overlay
            im3 = axes[2].imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            im4 = axes[2].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet', alpha=0.4)
            axes[2].set_title("Attention Overlay")
            axes[2].axis("off")

            fig.suptitle(
                f"{flavor_text} Sample {i} - Label: {'bonafide' if label == 0 else 'deepfake'} - Prediction: {'bonafide' if pred_label == 0 else 'deepfake'}",
                fontsize=16
            )

            # Save the figure
            save_path = os.path.join(save_dir, f"attention_visualization_{flavor_text}_{i}_label_{label}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()


    print(f"Enhanced visualizations saved to {save_dir}")

def attention_map_wrapper():

    #path = r"checkpoints/asvspoof-ast-model_TESTING_9_20250522_021137"
    path = r"checkpoints/asvspoof-pretrain-model_ADD_data_0_20250517_101737"

    ADD_DATASET_PATH = r"spectrograms/ADD"
    FOR_DATASET_PATH = r"spectrograms/FoR/for-2sec/for-2seconds"
    FOR_DATASET_PATH_TRAINING = r"spectrograms/FoR/for-2sec/for-2seconds/Training"
    FOR_DATASET_PATH_TESTING = r"spectrograms/FoR/for-2sec/for-2seconds/Testing"
    ASVS_DATASET_PATH = r"spectrograms"

    samples_add = {"genuine": 1000, "fake": 1000}
    samples_for = {"Real": 1000, "Fake": 1000}
    samples_asv = {"bonafide": 1000, "fake": 1000}
    ast = False

    model = load_pretrained_model_attention(
        saved_model_path=path,  # Original model name
        device="cuda"
    )

    dir = "attention-maps-all-TEST"
    embedding = 300
    """
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    """
    transform = transforms.Compose([
        StretchMelCropTime(224, 224),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    print("Loading datasets")
    asv_dataset, _, _ = load_ASV_dataset(path=ASVS_DATASET_PATH, samples=samples_asv, is_AST=ast, split=None,
                                         transform=transform, embedding_size=embedding)
    for_data = load_FOR_total(path=FOR_DATASET_PATH, samples=samples_for, is_AST=ast, transform=transform,
                              embedding_size=embedding)
    add_data, _, _ = load_ADD_dataset(path=ADD_DATASET_PATH, samples=samples_add, is_AST=ast, split=None,
                                      transform=transform, embedding_size=embedding)
    print("Making attention maps")
    samples = 10
    save_text = "ASV"
    generate_enhanced_attention_maps_pretrain(model, asv_dataset, num_samples=samples, flavor_text=save_text, save_dir=dir)
    save_text = "FoR"
    generate_enhanced_attention_maps_pretrain(model, for_data, num_samples=samples, flavor_text=save_text, save_dir=dir)
    save_text = "ADD"
    generate_enhanced_attention_maps_pretrain(model, add_data, num_samples=samples, flavor_text=save_text, save_dir=dir)

