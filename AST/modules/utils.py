import os

import librosa
import numpy as np


def get_input_and_labels(is_AST, batch, device):
    if is_AST:
        inputs = batch["input_values"].to(device)  # shape: (B, T, 128)
        labels = batch["labels"].to(device)
        return inputs, labels
    else:
        inputs, labels = batch  # batch is a tuple (inputs, labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels


def convert_to_spectrogram(file_path):
    sample_rate = 16000
    audio, sr = librosa.load(file_path, sr=sample_rate)

    window_size = int(0.025 * sr)  # 25ms window
    hop_length = int(0.01 * sr)  # 10ms hop size

    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                       hop_length=hop_length,
                                       win_length=window_size,
                                       window='hamming',
                                       power=2.0)

    return librosa.power_to_db(S, ref=np.max)

def save_spectrogram(file_path, save_path):
    spec = convert_to_spectrogram(file_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path + ".npy", spec)

def split_spectrogram_file(
    input_file,
    output_dir,
    target_frames=300,
    label="unlabeled",
    prefix=None,
    verbose=True
):
    """
    Splits a single spectrogram (.npy) into non-overlapping chunks and saves them.

    Args:
        input_file (str): Path to the input .npy spectrogram file (shape: [T, mel_bins]).
        output_dir (str): Directory where the output chunks will be saved.
        target_frames (int): Number of frames per chunk.
        label (str): Label to use for subfolder (e.g., "bonafide", "fake", "unlabeled").
        prefix (str): Optional prefix for naming output chunks. Defaults to input file name.
        verbose (bool): Whether to print status messages.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Load spectrogram
    spectrogram = np.load(input_file)
    spectrogram = spectrogram.T
    total_frames = spectrogram.shape[0]

    # How many full chunks we can make
    num_chunks = total_frames // target_frames

    if num_chunks == 0:
        if verbose:
            print(f"[Warning] File too short to split: {input_file} ({total_frames} frames)")
        return

    base_name = prefix or os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_chunks):
        start = i * target_frames
        end = start + target_frames
        chunk = spectrogram[start:end, :]  # shape: (target_frames, mel_bins)
        chunk = chunk.T
        output_path = os.path.join(label_dir, f"{base_name}_chunk{i}.npy")
        np.save(output_path, chunk)

        if verbose:
            print(f"Saved chunk {i+1}/{num_chunks} to {output_path}")
