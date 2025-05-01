import matplotlib.pyplot as plt
import numpy as np
import os

def plot_avg_audio_length(dataset, output_dir):
    lengths = {0: [], 1: []}  # 0 = bonafide, 1 = fake

    for file_path, label in dataset.files:
        spec = np.load(file_path)
        lengths[label].append(spec.shape[0])  # original number of frames before transpose

    avg_lengths = [np.mean(lengths[0]), np.mean(lengths[1])]
    plt.bar(["bonafide", "fake"], avg_lengths, color=["green", "red"])
    plt.ylabel("Average Length (Frames)")
    plt.title("Average Audio Length by Class")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "avg_audio_length.png"))
    plt.close()
