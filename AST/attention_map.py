import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import cv2

from Datasets import ASVspoofDataset
from benchmark_vson import load_modified_ast_model


def generate_attention_maps(model, dataset, num_samples=5, save_dir="attention-maps", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Generating attention maps"):
            sample = dataset[i]
            input_tensor = sample["input_values"].unsqueeze(0).to(device)  # shape: (1, 300, 128)
            label = sample["labels"].item()

            # Forward pass with attention
            outputs = model(
                input_values=input_tensor,
                output_attentions=True,
                return_dict=True
            )

            # Get last layer attention
            attn = outputs.attentions[-1]  # shape: (1, num_heads, seq_len, seq_len)
            attn = attn[0]  # (num_heads, seq_len, seq_len)

            # Average over all heads
            attn_avg = attn.mean(dim=0)  # shape: (seq_len, seq_len)

            # For classification tasks: Get CLS token attention to all other tokens
            cls_attn = attn_avg[0, 1:]  # CLS token's attention to other tokens
            attn_map_1d = cls_attn.cpu().numpy()

            # Get the actual sequence length of the model's tokens
            seq_len = attn_map_1d.shape[0]

            # Get spectrogram dimensions
            spec_height, spec_width = input_tensor.shape[1:3]  # Should be (300, 128)

            # Map attention to spectrogram dimensions more directly
            # Option 1: If there's a direct mapping between tokens and spectrogram patches
            # Calculate how many spectrogram pixels each token represents
            if hasattr(model, 'patch_size'):
                patch_size = model.patch_size  # If available from model
            else:
                # Estimate based on seq_len and spectrogram size
                # Assuming tokens are created by patching the spectrogram
                patches_h = int(np.ceil(spec_height / np.sqrt(seq_len)))
                patches_w = int(np.ceil(spec_width / np.sqrt(seq_len)))
                print(f"Estimated patch size: {patches_h} x {patches_w}")

            # Method 1: Try to reshape based on estimated mapping (better with known patch size)
            # This is a heuristic approach - ideally we would know the exact patching strategy
            h_tokens = int(np.ceil(spec_height / patches_h))
            w_tokens = int(np.ceil(spec_width / patches_w))

            # Handle case where seq_len doesn't match h_tokens * w_tokens exactly
            if seq_len < h_tokens * w_tokens:
                # Pad with zeros to match expected size
                attn_map_1d = np.pad(attn_map_1d, (0, h_tokens * w_tokens - seq_len))
            elif seq_len > h_tokens * w_tokens:
                # Truncate or resize
                attn_map_1d = attn_map_1d[:h_tokens * w_tokens]

            try:
                # Reshape to 2D grid matching token arrangement
                attn_map_2d = attn_map_1d.reshape(h_tokens, w_tokens)
                # Resize to match spectrogram dimensions
                attn_resized = cv2.resize(attn_map_2d, (spec_width, spec_height))
            except ValueError:
                # Fallback: If reshape fails, use your original approach
                print(
                    f"Warning: Could not reshape attention map to ({h_tokens}, {w_tokens}). Falling back to square reshape.")
                sqrt_len = int(np.ceil(np.sqrt(seq_len)))
                pad_len = sqrt_len ** 2 - seq_len
                attn_padded = np.pad(attn_map_1d, (0, pad_len), mode="constant")
                attn_square = attn_padded.reshape(sqrt_len, sqrt_len)
                attn_resized = cv2.resize(attn_square, (spec_width, spec_height))

            # Load original spectrogram
            spectrogram = sample["input_values"].cpu().numpy()  # shape: (300, 128)

            # Plot and overlay
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            ax.imshow(attn_resized, cmap='jet', alpha=0.4, origin='lower', aspect='auto')
            ax.set_title(f"Sample {i} - Label: {'bonafide' if label == 0 else 'spoof'}")
            ax.axis("off")

            save_path = os.path.join(save_dir, f"attention_map_{i}_label_{label}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()


def generate_enhanced_attention_maps(model, dataset, num_samples=5, save_dir="attention-maps-enhanced", device="cuda"):
    """
    Generate visualizations with three panels:
    1. Original spectrogram
    2. Attention map alone
    3. Overlaid visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Generating attention maps"):
            sample = dataset[i]
            input_tensor = sample["input_values"].unsqueeze(0).to(device)  # shape: (1, 300, 128)
            label = sample["labels"].item()

            # Forward pass with attention
            outputs = model(
                input_values=input_tensor,
                output_attentions=True,
                return_dict=True
            )

            # Get last layer attention
            attn = outputs.attentions[0]  # shape: (1, num_heads, seq_len, seq_len)
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

            # Plot 1: Original spectrogram
            im1 = axes[0].imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            axes[0].set_title("Original Spectrogram")
            axes[0].axis("off")
            fig.colorbar(im1, ax=axes[0], shrink=0.8)

            # Plot 2: Attention map alone
            # Normalize attention for better visualization
            attn_normalized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            im2 = axes[1].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet')
            axes[1].set_title("Attention Map")
            axes[1].axis("off")
            fig.colorbar(im2, ax=axes[1], shrink=0.8)

            # Plot 3: Overlay
            im3 = axes[2].imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            im4 = axes[2].imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet', alpha=0.4)
            axes[2].set_title("Attention Overlay")
            axes[2].axis("off")

            # Main title for the entire figure
            fig.suptitle(f"Sample {i} - Label: {'bonafide' if label == 0 else 'spoof'}", fontsize=16)

            # Save the figure
            save_path = os.path.join(save_dir, f"attention_visualization_{i}_label_{label}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()

            # Save individual images for closer inspection
            # Original spectrogram
            plt.figure(figsize=(8, 6))
            plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(shrink=0.8)
            plt.title(f"Original Spectrogram - Sample {i}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"orig_spec_{i}_label_{label}.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()

            # Attention map alone
            plt.figure(figsize=(8, 6))
            plt.imshow(attn_normalized, origin='lower', aspect='auto', cmap='jet')
            plt.colorbar(shrink=0.8)
            plt.title(f"Attention Map - Sample {i}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"attn_map_{i}_label_{label}.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()

            # Also save the attention values as numpy array for further analysis
            np.save(os.path.join(save_dir, f"attn_values_{i}_label_{label}.npy"), attn_resized)

    print(f"Enhanced visualizations saved to {save_dir}")

#Load model
path = r"checkpoints/asvspoof-ast-model15_100K_20250506_054106"
DATASET_PATH = r"spectrograms"
samples = {"bonafide": 10, "fake":10} # Load all

model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=path,      # Your saved model
    device="cuda"
)
dataset = ASVspoofDataset(data_dir=DATASET_PATH, max_per_class=samples)

generate_enhanced_attention_maps(model ,dataset, num_samples=10)