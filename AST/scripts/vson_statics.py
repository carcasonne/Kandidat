import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_CHECKPOINT = "/home/alsk/Kandidat/AST/checkpoints/asvspoof-ast-model15_100K_20250506_054106"
PRETRAIN_MODEL_CHECKPOINT = (
    "/home/alsk/Kandidat/AST/checkpoints/asvspoof-pretrain-model19_20250507_081555"
)
ADD_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/ADD"  
FOR_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"

ASVS_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms"
BATCH_SIZE = 16

AST_TARGET_FRAMES = 300

k_samples = {"bonafide": 1000, "fake": 1000}
twenty_k_samples = {"bonafide": 20000, "fake": 20000}
hundred_k_samples = {"bonafide": 100000, "fake": 100000}
all_samples = {"bonafide": 1000000000, "fake": 1000000000}