import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def collect_class_widths(dataset_path, class_names, max_width=5000):
    widths = {}
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found.")
            continue

        class_widths = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        data = np.load(file_path)
                        width = data.shape[-1]
                        if width <= max_width:
                            class_widths.append(width)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        widths[class_name.capitalize()] = class_widths
    return widths

def plot_boxplot(widths_dict, dataset_label, output_path):
    # Filter out any classes with empty data
    filtered = {label: values for label, values in widths_dict.items() if len(values) > 0}

    if len(filtered) < 2:
        print(f"⚠️ Warning: Not enough data to plot boxplot for {dataset_label}. Skipping.")
        return

    print(f"\n📊 Average widths for {dataset_label}:")
    for label, values in filtered.items():
        avg_width = np.mean(values)
        print(f"  {label}: {avg_width:.2f} samples (n={len(values)})")

    labels = list(filtered.keys())
    data = [filtered[label] for label in labels]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Boxplot of Spectrogram Widths – {dataset_label}")
    plt.xlabel("Class")
    plt.ylabel("Width")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot to {output_path}")



# Define paths to your datasets
dataset1_path = r"C:\Users\Askou\PycharmProjects\KIREPRO1PE\spectrograms\ADD"  # Contains 'fake' and 'genuine'
dataset2_path = r"C:\Users\Askou\PycharmProjects\KIREPRO1PE\spectrograms\FoR\for-2sec\for-2seconds\Training"  # Contains 'fake' and 'real'
dataset3_path = r"C:\Users\Askou\PycharmProjects\KIREPRO1PE\spectrograms\FoR\for-2sec\for-2seconds\Testing"  # Contains 'fake' and 'real'

# Dataset 1: fake + genuine
widths1 = collect_class_widths(dataset1_path, ['fake', 'genuine'])
plot_boxplot(widths1, "ADD", "box_plots/add_boxplot.png")

# Dataset 2: fake + real
widths2 = collect_class_widths(dataset2_path, ['fake', 'real'])
plot_boxplot(widths2, "FoR Training", "for_training.png")

# Dataset 2: fake + real
widths3 = collect_class_widths(dataset3_path, ['fake', 'real'])
plot_boxplot(widths3, "For Testing", "for_testing.png")