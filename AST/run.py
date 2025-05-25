from ADD_train_benchmark import *
from modules.utils import *

path = "in-the-wild/ssstwitter.com_1748187133462.mp3"
save_path = "in-the-wild/JD_vance_clip.npy"

#save_spectrogram(path, save_path)

#split_spectrogram_file(save_path, "spectrograms", 300, "fake")

gg = test_single_clip("spectrograms/fake")
