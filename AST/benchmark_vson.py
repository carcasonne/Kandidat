import os
import torch
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from transformers import ASTForAudioClassification

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTForAudioClassification
from Datasets import ASVspoofDataset, ADDdataset, FoRdataset
from wandb_login import login
import inspect

login()

# from your_dataset_module import ADDdataset

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "checkpoints/asvspoof-ast-model15_100K_20250506_054106"  # Replace with your actual saved path
ADD_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/ADD"  # Replace with your actual ADD dataset root
FOR_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms/FoR/for-2sec/for-2seconds"
ASVS_DATASET_PATH = "/home/alsk/Kandidat/AST/spectrograms"
BATCH_SIZE = 16
NUM_WORKERS = 4


def load_modified_ast_model(base_model_name, finetuned_model_path, device=None):
    """
    Load a model where only the last two layers are replaced with fine-tuned weights.

    Args:
        base_model_name: Name of the original pretrained model to start with
        finetuned_model_path: Path to the saved model with fine-tuned weights
        device: The device to load the model to ('cuda', 'cpu', or None to auto-detect)

    Returns:
        Model with base weights plus fine-tuned last layers
    """
    # Determine device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Loading base model {base_model_name}")
    # Start with the original pretrained model
    model = ASTForAudioClassification.from_pretrained(base_model_name)

    # Apply architecture modifications
    model.config.max_length = 300
    model.config.num_labels = 2
    model.config.id2label = {0: "bonafide", 1: "fake"}
    model.config.label2id = {"bonafide": 0, "fake": 1}

    # Modify the classifier for 2 classes (same as in your training code)
    if hasattr(model.classifier, 'dense'):
        model.classifier.dense = nn.Linear(model.classifier.dense.in_features, 2)
        if hasattr(model.classifier, 'out_proj'):
            model.classifier.out_proj = nn.Linear(2, 2)

    # Interpolate positional embeddings
    desired_max_length = 350
    position_embeddings = model.audio_spectrogram_transformer.embeddings.position_embeddings
    old_len = position_embeddings.shape[1]
    if old_len != desired_max_length:
        print(f"Interpolating position embeddings from {old_len} to {desired_max_length}")
        interpolated_pos_emb = F.interpolate(
            position_embeddings.permute(0, 2, 1),
            size=desired_max_length,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        model.audio_spectrogram_transformer.embeddings.position_embeddings = nn.Parameter(interpolated_pos_emb)

    # Load state dict from the fine-tuned model using safetensors format
    print(f"Loading fine-tuned weights from {finetuned_model_path}")

    try:
        # Try to load using safetensors
        from safetensors import safe_open
        from safetensors.torch import load_file

        safetensors_path = os.path.join(finetuned_model_path, "model.safetensors")
        print(f"safetensors path: {safetensors_path}")
        if os.path.exists(safetensors_path):
            print(f"Loading model from safetensors file: {safetensors_path}")
            finetuned_state_dict = load_file(safetensors_path)
        else:
            # Fall back to regular pytorch model loading
            pytorch_path = os.path.join(finetuned_model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_path):
                print(f"Loading model from PyTorch file: {pytorch_path}")
                finetuned_state_dict = torch.load(pytorch_path, map_location=device)
            else:
                # If neither file exists, try loading directly from the path
                print(f"Attempting to load model directly from: {finetuned_model_path}")
                model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)
                # Re-freeze the layers after loading
                N = 10
                for i in range(N):  # Layers 0 to 9
                    for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                        param.requires_grad = False
                model = model.to(device)
                print(f"Model successfully loaded directly")
                return model

    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        print("Trying alternate loading method...")

        # Try loading the model directly using from_pretrained
        try:
            print(f"Loading fine-tuned model directly from {finetuned_model_path}")
            model = ASTForAudioClassification.from_pretrained(finetuned_model_path, local_files_only=True)

            # Re-freeze the layers after loading
            N = 10
            for i in range(N):  # Layers 0 to 9
                for param in model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                    param.requires_grad = False

            model = model.to(device)
            print(f"Model successfully loaded directly")
            return model
        except Exception as e2:
            print(f"Failed to load model directly: {e2}")
            raise e2

    # If we got here, we have a state dict to filter
    # Filter the state dict to only include the last two transformer layers and classifier
    last_layers_dict = {}
    for key, value in finetuned_state_dict.items():
        # Include only the last two transformer layers (layers 10 and 11)
        if "encoder.layer.10." in key or "encoder.layer.11." in key:
            last_layers_dict[key] = value
        # Include classifier weights
        elif "classifier" in key:
            last_layers_dict[key] = value

    print(f"Selectively loading {len(last_layers_dict)} weights for the last layers and classifier")

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(finetuned_state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # Move the model to the specified device
    model = model.to(device)
    print(f"Model successfully prepared with fine-tuned last layers")

    return model

if __name__ == "__main__":
    # === Load the ADD dataset ===
    model = load_modified_ast_model(
        base_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",  # Original model name
        finetuned_model_path=MODEL_CHECKPOINT,      # Your saved model
        device="cuda"
    )

    samples = {"bonafide": 100000, "fake":100000} # Load all
    test_dataset = FoRdataset(data_dir=FOR_DATASET_PATH, max_per_class=samples)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # sanity check

    from collections import Counter
    label_counts = Counter(sample['labels'].item() for sample in test_dataset)
    print(f"Label distribution:\n{label_counts}")

    num_epochs = 1
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(test_loader, desc=f"Training Epoch {epoch}"):
            inputs = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"\n🧠 Epoch {epoch} Training Loss: {avg_loss:.4f}")

        # === Evaluation ===
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch}"):
                inputs = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"\n✅ Accuracy after Epoch {epoch}: {acc * 100:.2f}%")
        print(classification_report(all_labels, all_preds, target_names=["bonafide", "fake"]))

    # Final results
    print("\n🎉 Training complete. Final evaluation:")
    acc = accuracy_score(all_labels, all_preds)
    print(f"🏁 Final Accuracy: {acc * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["bonafide", "fake"]))
    # === Evaluation Metrics ===

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"TN: {tn}, FP: {fp} \n FN: {fn}, TP: {tp}")

    # === Weights & Biases Logging ===
    login()
    wandb.init(project="ADD Benchmark", entity="Holdet_thesis")

    wandb.log({
        "Accuracy": acc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn
    })


