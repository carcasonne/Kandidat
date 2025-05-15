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
            attn = outputs.attentions[-1]  # shape: (1, num_heads, num_tokens, num_tokens)
            attn = attn[0]  # (num_heads, num_tokens, num_tokens)

            # Average over all heads
            attn_avg = attn.mean(dim=0)  # shape: (num_tokens, num_tokens)

            cls_attn = attn_avg[0, 1:]  # (batch_size, num_tokens - 1)
            attn_map_1d = cls_attn.cpu().numpy()  # shape: (349,)

            # Rescale to spectrogram size directly (300x128)
            # We'll reshape to a square and then resize, or just interpolate to (300x128)

            # Method: reshape to a long vector → 2D → resize
            sqrt_len = int(np.ceil(np.sqrt(attn_map_1d.shape[0])))  # ~18.68 → 19
            pad_len = sqrt_len**2 - attn_map_1d.shape[0]            # pad with zeros
            attn_padded = np.pad(attn_map_1d, (0, pad_len), mode="constant")
            attn_square = attn_padded.reshape(sqrt_len, sqrt_len)

            # Resize to (300, 128) for overlay
            attn_resized = cv2.resize(attn_square, (128, 300))  # (H, W)

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


#Load model
path = r"checkpoints/asvspoof-ast-model0_20250513_172231"
DATASET_PATH = r"spectrograms"
samples = {"bonafide": 10, "fake":10} # Load all

model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=path,      # Your saved model
    device="cuda"
)
dataset = ASVspoofDataset(data_dir=DATASET_PATH, max_per_class=samples)


generate_attention_maps(model ,dataset, num_samples=5)