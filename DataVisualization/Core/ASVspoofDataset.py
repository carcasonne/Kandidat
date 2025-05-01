# Custom Dataset Class
class ASVspoofDataset(Dataset):
    def __init__(self, data_dir, max_per_class=50):
        self.data_dir = data_dir
        self.spec_dir = os.path.join(data_dir, "ASVSpoof")

        self.max_per_class = int(max_per_class) if max_per_class is not None else None

        self.class_map = {
            "bonafide": 0,
            "fake": 1
        }

        self.files = []
        for class_name, label in self.class_map.items():
            class_folder = os.path.join(self.spec_dir, class_name)
            class_files = [
                os.path.join(class_folder, file)
                for file in os.listdir(class_folder)
                if file.endswith(".npy")
            ]

            if self.max_per_class is not None:
                if self.max_per_class > len(class_files):
                    self.max_per_class = len(class_files)
                class_files = class_files[:self.max_per_class]

            self.files.extend([(file_path, label) for file_path in class_files])

        print(f"Loaded {len(self.files)} total spectrograms "
                f"({self.max_per_class if self.max_per_class else 'all'} per class)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]

        # Load precomputed log-mel spectrogram
        spectrogram = np.load(file_path).astype(np.float32)  # shape: (num_frames, 128)
        spectrogram = spectrogram.T

        # Ensure correct shape: (1024, 128)
        target_frames = 1024
        num_frames, num_mel_bins = spectrogram.shape

        if num_mel_bins != 128:
            raise ValueError(f"Expected 128 Mel bins, got {num_mel_bins} in file: {file_path}")

        if num_frames < target_frames:
            # Pad with zeros at the end
            pad_amount = target_frames - num_frames
            spectrogram = np.pad(spectrogram, ((0, pad_amount), (0, 0)), mode='constant')
        elif num_frames > target_frames:
            # Center crop
            start = (num_frames - target_frames) // 2
            spectrogram = spectrogram[start:start + target_frames, :]

        spectrogram = torch.tensor(spectrogram)  # shape: (1024, 128)

        return {
            "input_values": spectrogram,
            "labels": torch.tensor(label, dtype=torch.long)
        }