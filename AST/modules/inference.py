from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F



def run_inference(model, inference_dataset, device="cuda", batch_size=16, save_results=False, output_file="inference_results.csv"):
    model.eval()
    dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            print(f"shape of input = {inputs.shape}")

            outputs = model(input_values=inputs).logits
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for pred, true_label, prob in zip(preds.cpu(), labels.cpu(), probs.cpu()):
                results.append({
                    "true_label": "bonafide" if true_label.item() == 0 else "fake",
                    "prediction": "bonafide" if pred.item() == 0 else "fake",
                    "prob_bonafide": prob[0].item(),
                    "prob_fake": prob[1].item()
                })
    """
    if save_results:
        import csv
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {output_file}")
    """
    return results

def run_inference_pretrain(model, inference_dataset, device="cuda", batch_size=16, save_results=False, output_file="inference_results.csv"):
    model.eval()
    dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            inputs, labels = batch  # batch is a tuple (inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for pred, true_label, prob in zip(preds.cpu(), labels.cpu(), probs.cpu()):
                results.append({
                    "true_label": "bonafide" if true_label.item() == 0 else "fake",
                    "prediction": "bonafide" if pred.item() == 0 else "fake",
                    "prob_bonafide": prob[0].item(),
                    "prob_fake": prob[1].item()
                })
    """
    if save_results:
        import csv
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {output_file}")
    """
    return results


def print_inference_results(results, max_rows=20):
    """
    Nicely print the inference results with prediction correctness.
    """
    print("\n{:<5} {:<10} {:<12} {:<14} {:<14} {}".format(
        "Idx", "True", "Predicted", "Prob Bonafide", "Prob Fake", "Correct?"
    ))
    print("-" * 70)

    for i, res in enumerate(results[:max_rows]):
        correct = res["true_label"] == res["prediction"]
        correct_str = "✅" if correct else "❌"
        print("{:<5} {:<10} {:<12} {:<14.4f} {:<14.4f} {}".format(
            i,
            res["true_label"],
            res["prediction"],
            res["prob_bonafide"],
            res["prob_fake"],
            correct_str
        ))

    if len(results) > max_rows:
        print(f"... (showing {max_rows} of {len(results)} results)")

    # Optional: print summary stats
    total = len(results)
    correct = sum(1 for r in results if r["true_label"] == r["prediction"])
    acc = 100 * correct / total
    print(f"\n✅ Correct: {correct} / {total} ({acc:.2f}%)")