import os
import random

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def calc_attention_maps(model, flavor_text, device, train_dataset, samples, get_random = False):
    # Make sure model is in eval mode and on correct device
    model.eval()
    model.to(device)

    # Create directory to save attention maps
    save_dir = "attention_maps"
    os.makedirs(save_dir, exist_ok=True)

    sample_indices = []
    if get_random:
        for i in range(samples):
            idx = random.randint(0, len(train_dataset))
            sample_indices.append(idx)
    else:
        sample_indices = list(range(samples))

    # Load dataset without shuffling
    vis_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Loop through specific samples
    for idx, batch in enumerate(vis_loader):
        if idx not in sample_indices:
            continue

        inputs = batch["input_values"].to(device)  # Add batch dim if needed
        labels = batch["labels"].item()

        with torch.no_grad():
            outputs = model(input_values=inputs, output_attentions=True)
            attentions = outputs.attentions  # List of attention layers

        # Pick the last layer and average over all heads
        last_layer_attention = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
        avg_attention = last_layer_attention.mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

        # Plot and save heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(avg_attention, cmap="viridis")
        plt.title(f"Sample {idx} | Label: {labels} | Layer: Last | Heads: Averaged")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")

        filename = os.path.join(save_dir, f"{flavor_text}_attention_sample_{idx}_label_{labels}.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved attention map for sample {idx} to {filename}")