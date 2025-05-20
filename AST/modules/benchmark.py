from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

import modules.utils as utils
import modules.metrics as metrics


def benchmark(model, data_loader, flavor_text, is_AST, device, project_name):
    # === Benchmarking Loop ===
    all_preds = []
    all_labels = []
    all_probs = []  # Store raw probabilities for EER calculation
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Benchmarking"):
            inputs, labels = utils.get_input_and_labels(is_AST, batch, device)

            outputs = model(inputs)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            # Get class predictions (0 or 1)
            preds = torch.argmax(outputs, dim=1)

            # Get probabilities using softmax for EER calculation
            probs = F.softmax(outputs, dim=1)

            # Store spoof probability (class 1 probability) for EER calculation
            spoof_probs = probs[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(spoof_probs.cpu().numpy())

    # === Evaluation Metrics ===
    # Standard classification metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Benchmark Accuracy: {acc * 100:.2f}%")
    print(
        classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"])
    )

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate EER (Equal Error Rate)
    eer, threshold = metrics.compute_eer(all_labels, all_probs)
    print(f"🔍 Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"   Threshold at EER: {threshold:.4f}")

    # === Weights & Biases Logging ===
    wandb.login()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=project_name, entity="Holdet_thesis", name=flavor_text + "_" + date
    )

    wandb.log(
        {
            "Accuracy": acc,
            "True Positives": tp,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "EER": eer,
            "EER_Threshold": threshold,
        }
    )

    wandb.finish()

    return {
        "accuracy": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "eer": eer,
        "threshold": threshold,
    }
