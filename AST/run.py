from sys import prefix

from ADD_train_benchmark import *
from modules.utils import *

clip_name = "JD_vance_clip"

clip_names_real = ["dronning_marg", "hawk_tuah", "trump_inauguration"]
clip_names_fake = ["JD_vance_clip", "veo3_1_comedian", "veo3_2_street_interview", "veo3_3_natureguy", "veo3_5_interrogation", "veo3_6._sad_man",
                   "veo3_7_politician", "veo3_8_court", "veo3_9_movie_director", "veo3_10_oldguy", "veo3_11_comedian_v2" , "veo4_comedian_woman.mp3"]

AST_MODEL_CHECKPOINT = "checkpoints/AST_alltrain"
AST_model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    finetuned_model_path=AST_MODEL_CHECKPOINT,
    device="cuda",
)

for i in range(len(clip_names_real)):

    path = f"in-the-wild/{clip_names_real[i]}.mp3"
    save_path = f"{path}"
    # save_spectrogram(path, save_path)
    # split_spectrogram_file(f"{save_path}.npy", "spectrograms", 300, "real")
    print(f"Testing {save_path} ---------------------------------")
    gg = test_single_clip("spectrograms/real", clip_names_real[i], AST_model)

for i in range(len(clip_names_fake)):
    path = f"in-the-wild/{clip_names_fake[i]}.mp3"
    save_path = f"{path}"
    # save_spectrogram(path, save_path)
    # split_spectrogram_file(f"{save_path}.npy", "spectrograms", 300, "real")
    print(f"Testing {save_path} ---------------------------------")
    gg = test_single_clip("spectrograms/fake", clip_names_fake[i], AST_model)
