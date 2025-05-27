from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from tqdm import tqdm

import modules.utils as utils
import modules.metrics as metrics


def analyze_model_uncertainty(all_probs, all_labels, all_preds, uncertainty_threshold=0.1):
    """
    Analyze model uncertainty and confidence in predictions.
    
    Args:
        all_probs: Array of spoof probabilities (class 1 probabilities)
        all_labels: True labels
        all_preds: Predicted labels
        uncertainty_threshold: Threshold for considering a prediction uncertain
    
    Returns:
        Dictionary with uncertainty analysis results
    """
    # Calculate prediction confidence (distance from 0.5)
    confidence_scores = np.abs(np.array(all_probs) - 0.5)
    
    # Identify uncertain predictions (close to 0.5)
    uncertain_mask = confidence_scores < uncertainty_threshold
    
    # Analyze uncertain predictions
    uncertain_indices = np.where(uncertain_mask)[0]
    uncertain_accuracy = accuracy_score(
        np.array(all_labels)[uncertain_mask], 
        np.array(all_preds)[uncertain_mask]
    ) if len(uncertain_indices) > 0 else 0
    
    # Analyze confident predictions
    confident_mask = ~uncertain_mask
    confident_indices = np.where(confident_mask)[0]
    confident_accuracy = accuracy_score(
        np.array(all_labels)[confident_mask], 
        np.array(all_preds)[confident_mask]
    ) if len(confident_indices) > 0 else 0
    
    return {
        'total_uncertain': len(uncertain_indices),
        'uncertain_percentage': len(uncertain_indices) / len(all_probs) * 100,
        'uncertain_accuracy': uncertain_accuracy,
        'confident_accuracy': confident_accuracy,
        'uncertain_indices': uncertain_indices.tolist(),
        'confident_indices': confident_indices.tolist(),
        'avg_confidence': np.mean(confidence_scores),
        'confidence_scores': confidence_scores.tolist()
    }


def analyze_wrong_predictions(all_labels, all_preds, all_probs, data_loader=None):
    """
    Analyze wrongly classified samples in detail.
    
    Returns:
        Dictionary with detailed analysis of wrong predictions
    """
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Find wrong predictions
    wrong_mask = all_labels != all_preds
    wrong_indices = np.where(wrong_mask)[0]
    
    # Separate by class
    # False Positives: Predicted spoof (1) but actually bonafide (0)
    fp_mask = (all_preds == 1) & (all_labels == 0)
    fp_indices = np.where(fp_mask)[0]
    
    # False Negatives: Predicted bonafide (0) but actually spoof (1)  
    fn_mask = (all_preds == 0) & (all_labels == 1)
    fn_indices = np.where(fn_mask)[0]
    
    # Analyze confidence of wrong predictions
    wrong_probs = all_probs[wrong_mask]
    wrong_confidence = np.abs(wrong_probs - 0.5)
    
    fp_probs = all_probs[fp_mask] if len(fp_indices) > 0 else []
    fn_probs = all_probs[fn_mask] if len(fn_indices) > 0 else []
    
    return {
        'total_wrong': len(wrong_indices),
        'wrong_percentage': len(wrong_indices) / len(all_labels) * 100,
        'false_positive_indices': fp_indices.tolist(),
        'false_negative_indices': fn_indices.tolist(),
        'fp_count': len(fp_indices),
        'fn_count': len(fn_indices),
        'wrong_avg_confidence': np.mean(wrong_confidence) if len(wrong_confidence) > 0 else 0,
        'fp_avg_prob': np.mean(fp_probs) if len(fp_probs) > 0 else 0,
        'fn_avg_prob': np.mean(fn_probs) if len(fn_probs) > 0 else 0,
        'wrong_indices': wrong_indices.tolist()
    }


def create_detailed_confusion_matrix(all_labels, all_preds, save_path=None):
    """
    Create and optionally save a detailed confusion matrix visualization.
    """
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bonafide', 'Spoof'],
                yticklabels=['Bonafide', 'Spoof'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm


def analyze_probability_distribution(all_probs, all_labels, save_path=None):
    """
    Analyze and visualize the distribution of prediction probabilities.
    """
    bonafide_probs = np.array(all_probs)[np.array(all_labels) == 0]
    spoof_probs = np.array(all_probs)[np.array(all_labels) == 1]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of probabilities
    plt.subplot(1, 2, 1)
    plt.hist(bonafide_probs, bins=50, alpha=0.7, label='Bonafide (True)', color='blue')
    plt.hist(spoof_probs, bins=50, alpha=0.7, label='Spoof (True)', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.xlabel('Spoof Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    
    # Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [bonafide_probs, spoof_probs]
    plt.boxplot(data_to_plot, labels=['Bonafide', 'Spoof'])
    plt.ylabel('Spoof Probability')
    plt.title('Probability Distribution by True Class')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved to {save_path}")
    
    plt.close()
    
    return {
        'bonafide_mean_prob': np.mean(bonafide_probs),
        'bonafide_std_prob': np.std(bonafide_probs),
        'spoof_mean_prob': np.mean(spoof_probs),
        'spoof_std_prob': np.std(spoof_probs)
    }


def calculate_additional_metrics(all_labels, all_probs):
    """
    Calculate additional evaluation metrics like AUC-ROC, AUC-PR, etc.
    """
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }


def benchmark(model, data_loader, flavor_text, is_AST, device, project_name, 
              save_plots=True, uncertainty_threshold=0.1):
    """
    Enhanced benchmarking function with comprehensive error analysis.
    """
    print(f"\n🚀 Starting enhanced benchmark: {flavor_text}")
    
    # Create output directory for plots
    output_dir = f"benchmark_results/{flavor_text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {output_dir}")
    
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

    print(f"✅ Processed {len(all_labels)} samples")

    # === Basic Evaluation Metrics ===
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Benchmark Accuracy: {acc * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"]))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate EER (Equal Error Rate)
    eer, threshold = metrics.compute_eer(all_labels, all_probs)
    print(f"🔍 Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"   Threshold at EER: {threshold:.4f}")

    # === Enhanced Analysis ===
    print("\n" + "="*50)
    print("🔬 ENHANCED ANALYSIS")
    print("="*50)
    
    # 1. Confusion Matrix Analysis
    if save_plots:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        create_detailed_confusion_matrix(all_labels, all_preds, cm_path)
    else:
        create_detailed_confusion_matrix(all_labels, all_preds)
    
    # 2. Model Uncertainty Analysis
    uncertainty_analysis = analyze_model_uncertainty(
        all_probs, all_labels, all_preds, uncertainty_threshold
    )
    print(f"\n🎯 UNCERTAINTY ANALYSIS:")
    print(f"   Uncertain predictions: {uncertainty_analysis['total_uncertain']} ({uncertainty_analysis['uncertain_percentage']:.2f}%)")
    print(f"   Accuracy on uncertain samples: {uncertainty_analysis['uncertain_accuracy']:.3f}")
    print(f"   Accuracy on confident samples: {uncertainty_analysis['confident_accuracy']:.3f}")
    print(f"   Average confidence: {uncertainty_analysis['avg_confidence']:.3f}")
    
    # 3. Wrong Predictions Analysis
    wrong_analysis = analyze_wrong_predictions(all_labels, all_preds, all_probs)
    print(f"\n❌ WRONG PREDICTIONS ANALYSIS:")
    print(f"   Total wrong: {wrong_analysis['total_wrong']} ({wrong_analysis['wrong_percentage']:.2f}%)")
    print(f"   False Positives (bonafide → spoof): {wrong_analysis['fp_count']}")
    print(f"   False Negatives (spoof → bonafide): {wrong_analysis['fn_count']}")
    print(f"   Average confidence of wrong predictions: {wrong_analysis['wrong_avg_confidence']:.3f}")
    print(f"   Average spoof prob for FPs: {wrong_analysis['fp_avg_prob']:.3f}")
    print(f"   Average spoof prob for FNs: {wrong_analysis['fn_avg_prob']:.3f}")
    
    # 4. Probability Distribution Analysis
    if save_plots:
        prob_dist_path = os.path.join(output_dir, "probability_distribution.png")
        prob_stats = analyze_probability_distribution(all_probs, all_labels, prob_dist_path)
    else:
        prob_stats = analyze_probability_distribution(all_probs, all_labels)
    
    print(f"\n📊 PROBABILITY DISTRIBUTION:")
    print(f"   Bonafide mean prob: {prob_stats['bonafide_mean_prob']:.3f} ± {prob_stats['bonafide_std_prob']:.3f}")
    print(f"   Spoof mean prob: {prob_stats['spoof_mean_prob']:.3f} ± {prob_stats['spoof_std_prob']:.3f}")
    
    # 5. Additional Metrics
    additional_metrics = calculate_additional_metrics(all_labels, all_probs)
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   ROC-AUC: {additional_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {additional_metrics['pr_auc']:.4f}")

    # === Detailed Lists of Wrong Cases ===
    print(f"\n📝 DETAILED WRONG CASE INDICES:")
    print(f"   False Positive indices (first 20): {wrong_analysis['false_positive_indices'][:20]}")
    print(f"   False Negative indices (first 20): {wrong_analysis['false_negative_indices'][:20]}")
    print(f"   Most uncertain indices (first 20): {uncertainty_analysis['uncertain_indices'][:20]}")

    # === Weights & Biases Logging ===
    wandb.login()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=project_name, 
        entity="Holdet_thesis", 
        name=flavor_text + "_" + date, 
        mode="offline"
    )

    # Log all metrics to wandb
    wandb_metrics = {
        # Basic metrics
        "Accuracy": acc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "EER": eer,
        "EER_Threshold": threshold,
        
        # Additional metrics
        "ROC_AUC": additional_metrics['roc_auc'],
        "PR_AUC": additional_metrics['pr_auc'],
        
        # Uncertainty metrics
        "Uncertain_Count": uncertainty_analysis['total_uncertain'],
        "Uncertain_Percentage": uncertainty_analysis['uncertain_percentage'],
        "Uncertain_Accuracy": uncertainty_analysis['uncertain_accuracy'],
        "Confident_Accuracy": uncertainty_analysis['confident_accuracy'],
        "Avg_Confidence": uncertainty_analysis['avg_confidence'],
        
        # Wrong prediction metrics
        "Wrong_Total": wrong_analysis['total_wrong'],
        "Wrong_Percentage": wrong_analysis['wrong_percentage'],
        "FP_Count": wrong_analysis['fp_count'],
        "FN_Count": wrong_analysis['fn_count'],
        "Wrong_Avg_Confidence": wrong_analysis['wrong_avg_confidence'],
        "FP_Avg_Prob": wrong_analysis['fp_avg_prob'],
        "FN_Avg_Prob": wrong_analysis['fn_avg_prob'],
        
        # Probability distribution stats
        "Bonafide_Mean_Prob": prob_stats['bonafide_mean_prob'],
        "Bonafide_Std_Prob": prob_stats['bonafide_std_prob'],
        "Spoof_Mean_Prob": prob_stats['spoof_mean_prob'],
        "Spoof_Std_Prob": prob_stats['spoof_std_prob'],
    }
    
    wandb.log(wandb_metrics)
    
    # Log plots to wandb if they exist
    if save_plots:
        try:
            wandb.log({
                "confusion_matrix": wandb.Image(os.path.join(output_dir, "confusion_matrix.png")),
                "probability_distribution": wandb.Image(os.path.join(output_dir, "probability_distribution.png"))
            })
        except Exception as e:
            print(f"Warning: Could not log images to wandb: {e}")

    wandb.finish()

    # === Return comprehensive results ===
    results = {
        # Basic metrics
        "accuracy": acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "eer": eer, "threshold": threshold,
        
        # Enhanced analysis results
        "uncertainty_analysis": uncertainty_analysis,
        "wrong_analysis": wrong_analysis,
        "probability_stats": prob_stats,
        "additional_metrics": additional_metrics,
        
        # Output directory
        "output_dir": output_dir if save_plots else None
    }
    
    print(f"\n✅ Enhanced benchmark completed! Results {'saved to ' + output_dir if save_plots else 'generated'}")
    
    return results