import matplotlib.pyplot as plt
import numpy as np
import os

def plot_hist_audio_length(dataset, output_dir):
    lengths = {0: [], 1: []}

    for file_path, label in dataset.files:
        spec = np.load(file_path)
        lengths[label].append(spec.shape[0])

    plt.hist(lengths[0], bins=30, alpha=0.7, label="bonafide", color="green")
    plt.hist(lengths[1], bins=30, alpha=0.7, label="fake", color="red")
    plt.xlabel("Length (Frames)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Audio Lengths by Class")
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "hist_audio_length.png"))
    plt.close()
