from sys import prefix

from ADD_train_benchmark import *
from modules.utils import *

clip_name = "JD_vance_clip"

path = f"in-the-wild/{clip_name}.mp3"
save_path = f"{path}"

#save_spectrogram(path, save_path)

#split_spectrogram_file(f"{save_path}.npy", "spectrograms", 300, "real")

gg = test_single_clip_pretrain("spectrograms/fake", clip_name)
