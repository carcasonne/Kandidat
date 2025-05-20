import torch
from Datasets import (
    ADDdataset,
    ADDdatasetPretrain,
    ASVspoofDataset,
    FoRdataset,
    FoRdatasetPretrain,
)
from torch.utils.data import DataLoader
from torchvision import transforms

import modules.benchmark as benchmark
import modules.models as models

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AST_MODEL_CHECKPOINT = "/home/alsk/Kandidat/AST/checkpoints/asvspoof-ast-model15_100K_20250506_054106"  # Replace with your actual saved path
PRETRAIN_MODEL_CHECKPOINT = (
    "/home/alsk/Kandidat/AST/checkpoints/asvspoof-pretrain-model19_20250507_081555"
)
ADD_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/ADD"  # Replace with your actual ADD dataset root
FOR_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
ASVS_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms"
BATCH_SIZE = 16

if __name__ == "__main__":
    print("Yo its me from benchmark")
    # === Load the ADD dataset ===
    AST_model = models.load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        finetuned_model_path=AST_MODEL_CHECKPOINT,
        device="cuda",
    )
    Pretrain_model = models.load_pretrained_model(saved_model_path=PRETRAIN_MODEL_CHECKPOINT, device=DEVICE)
    base_AST_model = models.load_base_ast_model(device=DEVICE)

    samples = {"bonafide": 100000, "fake": 100000}
    asv_samples = {"bonafide": 100000, "fake": 100000}

    # AST Datasets
    ast_target_frames = 300  # why 300 you ask? i dont fucking know, i answer
    add_test_dataset = ADDdataset(
        data_dir=ADD_DATASET_PATH,
        max_per_class=samples,
        target_frames=ast_target_frames,
    )
    add_test_loader = DataLoader(add_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for_test_dataset = FoRdataset(
        data_dir=FOR_DATASET_PATH,
        max_per_class=samples,
        target_frames=ast_target_frames,
    )
    for_test_loader = DataLoader(for_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    asvs_test_dataset = ASVspoofDataset(
        data_dir=ASVS_DATASET_PATH,
        max_per_class=asv_samples,
        target_frames=ast_target_frames,
    )
    asvs_test_loader = DataLoader(
        asvs_test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Pretrain datasets
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )

    pre_add_test_dataset = ADDdatasetPretrain(
        data_dir=ADD_DATASET_PATH, max_per_class=samples, transform=transform
    )
    pre_add_test_loader = DataLoader(
        pre_add_test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    pre_for_test_dataset = FoRdatasetPretrain(
        data_dir=FOR_DATASET_PATH, max_per_class=samples, transform=transform
    )
    pre_for_test_loader = DataLoader(
        pre_for_test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    run_name_1 = "Sanity_check"
    benchmark.benchmark(AST_model, asvs_test_loader, run_name_1, True, DEVICE, "ASV-Benchmark-Sanity-Check-v2")

    # run_name_2 = "Sanity_check_base"
    # benchmark.benchmark(base_AST_model, asvs_test_loader, run_name_2, True, DEVICE)

    # run_name = "AST_benchmark_ADD"
    # benchmark.benchmark(AST_model, add_test_loader, run_name, True, DEVICE)

    # run_name1 = "AST_benchmark_FoR"
    # benchmark.benchmark(AST_model, for_test_loader, run_name1, True, DEVICE)

    # run_name2 = "Pretrain_benchmark_ADD"
    # benchmark.benchmark(Pretrain_model, pre_add_test_loader, run_name2, False, DEVICE)

    # run_name3 = "Pretrain_benchmark_FoR"
    # benchmark.benchmark(Pretrain_model, pre_for_test_loader, run_name3, False, DEVICE)
