import os
import numpy as np
import matplotlib.pyplot as plt

# Directories where the spectrograms are stored
fake_dir = 'spectrograms/ASVSpoof/fake'        # Replace with your actual path
bonafide_dir = 'spectrograms/ASVSpoof/bonafide'  # Replace with your actual path
save_dir = 'figures'

os.makedirs(save_dir, exist_ok=True)
# Function to extract shapes from files in a directory
def get_spectrogram_shapes(directory):
    shapes = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                spectrogram = np.load(file_path)
                shapes.append(spectrogram.shape)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return shapes

# Get the shapes for 'fake' and 'bonafide' spectrograms
print("Processing 'fake' spectrograms...")
fake_shapes = get_spectrogram_shapes(fake_dir)

print("Processing 'bonafide' spectrograms...")
bonafide_shapes = get_spectrogram_shapes(bonafide_dir)

# Compute average shapes
avg_fake_shape = np.mean(fake_shapes, axis=0)
avg_bonafide_shape = np.mean(bonafide_shapes, axis=0)

# Print the average shapes
print(f"Average shape of 'fake' spectrograms: {avg_fake_shape}")
print(f"Average shape of 'bonafide' spectrograms: {avg_bonafide_shape}")

# Prepare data for boxplots
fake_heights = [shape[0] for shape in fake_shapes]
fake_widths = [shape[1] for shape in fake_shapes]
bonafide_heights = [shape[0] for shape in bonafide_shapes]
bonafide_widths = [shape[1] for shape in bonafide_shapes]

# Create and save the height boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([fake_heights, bonafide_heights], labels=['Fake Heights', 'Bonafide Heights'])
plt.title('Boxplot of Spectrogram Heights')
plt.ylabel('Height')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'spectrogram_width_boxplot.png'))
plt.close()

# Create and save the width boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([fake_widths, bonafide_widths], labels=['Fake Widths', 'Bonafide Widths'])
plt.title('Boxplot of Spectrogram Widths')
plt.ylabel('Width')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'spectrogram_width_boxplot.png'))
plt.close()
