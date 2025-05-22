import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import cv2
from attention_map import generate_enhanced_attention_maps
from torchvision import datasets, transforms
from Datasets import *
from modules.models import load_modified_ast_model
# Load model
path = r"checkpoints/asvspoof-ast-model_TESTING_9_20250522_021137"

ADD_DATASET_PATH = r"spectrograms/ADD"
FOR_DATASET_PATH = r"spectrograms/FoR/for-2sec/for-2seconds"
FOR_DATASET_PATH_TRAINING = r"spectrograms/FoR/for-2sec/for-2seconds/Training"
FOR_DATASET_PATH_TESTING = r"spectrograms/FoR/for-2sec/for-2seconds/Testing"
ASVS_DATASET_PATH = r"spectrograms"

samples_add = {"genuine": 50000, "fake":50000}
samples_for = {"Real": 100000, "Fake":100000}
samples_asv = {"bonafide": 50000, "fake":50000}

model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=path,      # Your saved model
    device="cuda"
)

dir = "attention-maps-all-train"
embedding = 300
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

asv_dataset, _, _ = load_ASV_dataset(path=ASVS_DATASET_PATH, samples=samples_asv, is_AST=True, split=None, transform=transform, embedding_size=embedding)
for_data = load_FOR_total(path=FOR_DATASET_PATH, samples=samples_for, is_AST=True, transform=transform, embedding_size=embedding)
add_data, _, _ = load_ADD_dataset(path=ADD_DATASET_PATH, samples=samples_add, is_AST=True, split=None, transform=transform, embedding_size=embedding)

save_text = "ASV"
generate_enhanced_attention_maps(model ,asv_dataset, num_samples=10, flavor_text=save_text, save_dir=dir)
save_text = "FoR"
generate_enhanced_attention_maps(model ,for_data, num_samples=10, flavor_text=save_text, save_dir=dir)
save_text = "ADD"
generate_enhanced_attention_maps(model ,add_data, num_samples=10, flavor_text=save_text, save_dir=dir)