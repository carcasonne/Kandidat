import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import cv2
from attention_map import generate_enhanced_attention_maps

from Datasets import ASVspoofDataset
from benchmark_vson import load_modified_ast_model
# Load model
path = r"checkpoints/asvspoof-ast-model15_100K_20250506_054106"
DATASET_PATH = r"spectrograms"
samples = {"bonafide": 10, "fake" :10} # Load all

model = load_modified_ast_model(
    base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
    finetuned_model_path=path,      # Your saved model
    device="cuda"
)
dataset = ASVspoofDataset(data_dir=DATASET_PATH, max_per_class=samples)

generate_enhanced_attention_maps(model ,dataset, num_samples=10)