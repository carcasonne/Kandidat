import matplotlib.pyplot as plt
import os
from collections import Counter

def plot_class_balance(dataset, output_dir):
    label_counts = Counter(label for _, label in dataset.files)
    class_names = ["bonafide", "fake"]
    counts = [label_counts[0], label_counts[1]]

    plt.bar(class_names, counts, color=['green', 'red'])
    plt.title("Class Balance")
    plt.ylabel("Number of Samples")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "class_balance.png"))
    plt.close()
