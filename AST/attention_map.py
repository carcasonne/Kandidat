import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

def calc_attention_maps(model, flavor_text, device, train_dataset, samples, get_random=False):
    model.eval()
    model.to(device)

    save_dir = "attention_maps"
    os.makedirs(save_dir, exist_ok=True)

    sample_indices = random.sample(range(len(train_dataset)), samples) if get_random else list(range(samples))

    vis_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(vis_loader):
        if idx not in sample_indices:
            continue

        inputs = batch["input_values"].to(device)
        labels = batch["labels"].item()

        # If you have original images, get them (assuming they are in batch["image"])
        original_image = batch["input_values"][0]  # (C, H, W) format
        if isinstance(original_image, torch.Tensor):
            original_image = to_pil_image(original_image)

        with torch.no_grad():
            outputs = model(input_values=inputs, output_attentions=True)
            attentions = outputs.attentions  # List of attention layers

        # Take the attention from the last layer, average over heads
        last_layer_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        avg_attention = last_layer_attention.mean(dim=0)  # (seq_len, seq_len)

        # Depending on your model, you might need to select the [CLS] token's attention
        # e.g., avg_attention = avg_attention[0, 1:] if it includes CLS token.

        # For visualization, you often take the attention of the [CLS] token to all patches
        attention_to_patches = avg_attention[0, 1:]  # Assuming first token (index 0) is [CLS]

        # Reshape the attention to image-like grid
        num_patches = attention_to_patches.shape[0]
        grid_size = int(np.sqrt(num_patches))
        attention_map = attention_to_patches.reshape(grid_size, grid_size).cpu().numpy()

        # Normalize the attention map
        attention_map -= attention_map.min()
        attention_map /= attention_map.max()

        # Resize attention map to match original image size
        attention_map = torch.tensor(attention_map)
        attention_map = transforms.Resize((original_image.height, original_image.width))(attention_map.unsqueeze(0)).squeeze(0).numpy()

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.imshow(attention_map, cmap='jet', alpha=0.5)  # alpha controls transparency
        plt.axis('off')
        plt.title(f"Sample {idx} | Label: {labels}")

        filename = os.path.join(save_dir, f"{flavor_text}_attention_sample_{idx}_label_{labels}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        print(f"Saved attention overlay map for sample {idx} to {filename}")
