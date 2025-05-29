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
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef
)
from tqdm import tqdm

import modules.utils as utils
import modules.metrics as metrics
import modules.attention_map as attention_map


def print_confusion_matrix(cm, class_names=['Bonafide', 'Spoof']):
    """
    Print a nicely formatted confusion matrix in the terminal.
    """
    print("\n" + "="*40)
    print("📊 CONFUSION MATRIX")
    print("="*40)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Print header
    print(f"{'':>12} {'Predicted':>20}")
    print(f"{'Actual':>12} {class_names[0]:>10} {class_names[1]:>10} {'Total':>8}")
    print("-" * 45)
    
    # Print rows
    for i, class_name in enumerate(class_names):
        row_total = cm[i].sum()
        print(f"{class_name:>10} {cm[i][0]:>8} ({cm_percent[i][0]:>5.1f}%) "
              f"{cm[i][1]:>8} ({cm_percent[i][1]:>5.1f}%) {row_total:>6}")
    
    # Print totals
    col_totals = cm.sum(axis=0)
    total = cm.sum()
    print("-" * 45)
    print(f"{'Total':>10} {col_totals[0]:>10} {col_totals[1]:>10} {total:>6}")
    
    # Print key metrics derived from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n📈 Key Rates:")
    print(f"   True Positive Rate (Sensitivity/Recall): {sensitivity:.3f}")
    print(f"   True Negative Rate (Specificity): {specificity:.3f}")
    print(f"   False Positive Rate: {fp/(fp+tn):.3f}")
    print(f"   False Negative Rate: {fn/(fn+tp):.3f}")


def calculate_comprehensive_metrics(all_labels, all_preds, all_probs):
    """
    Calculate a comprehensive set of evaluation metrics.
    """
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Basic classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = _[optimal_idx] if len(_) > optimal_idx else 0.5
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'matthews_corrcoef': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'youden_j': j_scores[optimal_idx] if len(j_scores) > optimal_idx else 0
    }


def analyze_threshold_performance(all_labels, all_probs, thresholds=None):
    """
    Analyze model performance across different decision thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.1)
    
    results = []
    
    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })
    
    return results


def analyze_class_distribution(all_labels, all_probs):
    """
    Analyze the distribution and characteristics of each class.
    """
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Class counts
    bonafide_count = np.sum(all_labels == 0)
    spoof_count = np.sum(all_labels == 1)
    total = len(all_labels)
    
    # Probability statistics by class
    bonafide_probs = all_probs[all_labels == 0]
    spoof_probs = all_probs[all_labels == 1]
    
    return {
        'bonafide_count': bonafide_count,
        'spoof_count': spoof_count,
        'bonafide_percentage': bonafide_count / total * 100,
        'spoof_percentage': spoof_count / total * 100,
        'bonafide_prob_stats': {
            'mean': np.mean(bonafide_probs),
            'std': np.std(bonafide_probs),
            'median': np.median(bonafide_probs),
            'min': np.min(bonafide_probs),
            'max': np.max(bonafide_probs),
            'q25': np.percentile(bonafide_probs, 25),
            'q75': np.percentile(bonafide_probs, 75)
        },
        'spoof_prob_stats': {
            'mean': np.mean(spoof_probs),
            'std': np.std(spoof_probs),
            'median': np.median(spoof_probs),
            'min': np.min(spoof_probs),
            'max': np.max(spoof_probs),
            'q25': np.percentile(spoof_probs, 25),
            'q75': np.percentile(spoof_probs, 75)
        }
    }


def analyze_prediction_calibration(all_labels, all_probs, n_bins=10):
    """
    Analyze how well-calibrated the model's probability predictions are.
    """
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_results = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_probs > bin_lower) & (all_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = all_labels[in_bin].mean()
            avg_confidence_in_bin = all_probs[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            calibration_results.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': count_in_bin,
                'proportion': prop_in_bin
            })
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0
    for result in calibration_results:
        ece += result['proportion'] * abs(result['accuracy'] - result['confidence'])
    
    return {
        'calibration_bins': calibration_results,
        'expected_calibration_error': ece
    }


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
              save_plots=True, uncertainty_threshold=0.1, num_attention_samples=5):
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
    all_inputs = [] # Store inputs for attention map generation
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
            all_inputs.extend(inputs.cpu())

    print(f"✅ Processed {len(all_labels)} samples")

    # === Basic Evaluation Metrics ===
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Benchmark Accuracy: {acc * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["bonafide", "spoof"]))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Print confusion matrix in terminal
    print_confusion_matrix(cm)

    # Calculate EER (Equal Error Rate)
    eer, threshold = metrics.compute_eer(all_labels, all_probs)
    print(f"\n🔍 Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"   Threshold at EER: {threshold:.4f}")

    # === Enhanced Analysis ===
    print("\n" + "="*60)
    print("🔬 COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # 1. Comprehensive metrics
    comp_metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    print("\n📊 COMPREHENSIVE METRICS:")
    print(f"   Accuracy: {comp_metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {comp_metrics['balanced_accuracy']:.4f}")
    print(f"   Precision: {comp_metrics['precision']:.4f}")
    print(f"   Recall: {comp_metrics['recall']:.4f}")
    print(f"   F1-Score: {comp_metrics['f1_score']:.4f}")
    print(f"   Matthews Correlation Coefficient: {comp_metrics['matthews_corrcoef']:.4f}")
    print(f"   ROC-AUC: {comp_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {comp_metrics['pr_auc']:.4f}")
    print(f"   Optimal Threshold (Youden's J): {comp_metrics['optimal_threshold']:.4f}")
    
    # 2. Class distribution analysis
    class_dist = analyze_class_distribution(all_labels, all_probs)
    print("\n🎯 CLASS DISTRIBUTION:")
    print(f"   Bonafide samples: {class_dist['bonafide_count']} ({class_dist['bonafide_percentage']:.1f}%)")
    print(f"   Spoof samples: {class_dist['spoof_count']} ({class_dist['spoof_percentage']:.1f}%)")
    print(f"   Dataset balance ratio: {class_dist['bonafide_count']/class_dist['spoof_count']:.2f}:1")
    
    print("\n📈 PROBABILITY STATISTICS BY CLASS:")
    b_stats = class_dist['bonafide_prob_stats']
    s_stats = class_dist['spoof_prob_stats']
    print(f"   Bonafide - Mean: {b_stats['mean']:.3f}, Std: {b_stats['std']:.3f}, Median: {b_stats['median']:.3f}")
    print(f"   Spoof    - Mean: {s_stats['mean']:.3f}, Std: {s_stats['std']:.3f}, Median: {s_stats['median']:.3f}")
    print(f"   Class Separation (mean diff): {abs(s_stats['mean'] - b_stats['mean']):.3f}")
    
    # 3. Model calibration analysis
    calibration = analyze_prediction_calibration(all_labels, all_probs)
    print("\n🎚️ MODEL CALIBRATION:")
    print(f"   Expected Calibration Error (ECE): {calibration['expected_calibration_error']:.4f}")
    print("   Calibration by confidence bins:")
    for bin_info in calibration['calibration_bins']:
        print(f"     [{bin_info['bin_lower']:.1f}-{bin_info['bin_upper']:.1f}]: "
              f"Accuracy={bin_info['accuracy']:.3f}, Confidence={bin_info['confidence']:.3f}, "
              f"Count={bin_info['count']}")
    
    # 4. Threshold analysis
    threshold_analysis = analyze_threshold_performance(all_labels, all_probs)
    print("\n⚖️ THRESHOLD PERFORMANCE ANALYSIS:")
    print("   Threshold | Accuracy | Precision | Recall | F1-Score")
    print("   " + "-"*50)
    for result in threshold_analysis:
        print(f"      {result['threshold']:.1f}    |   {result['accuracy']:.3f}  |   {result['precision']:.3f}   | {result['recall']:.3f}  |  {result['f1_score']:.3f}")
    
    # 5. Model Uncertainty Analysis
    uncertainty_analysis = analyze_model_uncertainty(
        all_probs, all_labels, all_preds, uncertainty_threshold
    )
    print("\n🎯 UNCERTAINTY ANALYSIS:")
    print(f"   Uncertain predictions: {uncertainty_analysis['total_uncertain']} ({uncertainty_analysis['uncertain_percentage']:.2f}%)")
    print(f"   Accuracy on uncertain samples: {uncertainty_analysis['uncertain_accuracy']:.3f}")
    print(f"   Accuracy on confident samples: {uncertainty_analysis['confident_accuracy']:.3f}")
    print(f"   Average confidence: {uncertainty_analysis['avg_confidence']:.3f}")
    
    # 6. Wrong Predictions Analysis
    wrong_analysis = analyze_wrong_predictions(all_labels, all_preds, all_probs)
    print("\n❌ WRONG PREDICTIONS ANALYSIS:")
    print(f"   Total wrong: {wrong_analysis['total_wrong']} ({wrong_analysis['wrong_percentage']:.2f}%)")
    print(f"   False Positives (bonafide → spoof): {wrong_analysis['fp_count']}")
    print(f"   False Negatives (spoof → bonafide): {wrong_analysis['fn_count']}")
    print(f"   Average confidence of wrong predictions: {wrong_analysis['wrong_avg_confidence']:.3f}")
    print(f"   Average spoof prob for FPs: {wrong_analysis['fp_avg_prob']:.3f}")
    print(f"   Average spoof prob for FNs: {wrong_analysis['fn_avg_prob']:.3f}")
    
    # 7. Visualizations (optional)    
    if save_plots:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        create_detailed_confusion_matrix(all_labels, all_preds, cm_path)
        
        prob_dist_path = os.path.join(output_dir, "probability_distribution.png")
        prob_stats = analyze_probability_distribution(all_probs, all_labels, prob_dist_path)
    else:
        create_detailed_confusion_matrix(all_labels, all_preds)
        prob_stats = analyze_probability_distribution(all_probs, all_labels)

    # === Detailed Lists of Wrong Cases ===
    print("\n📝 DETAILED WRONG CASE INDICES:")
    print(f"   False Positive indices (first 20): {wrong_analysis['false_positive_indices'][:20]}")
    print(f"   False Negative indices (first 20): {wrong_analysis['false_negative_indices'][:20]}")
    print(f"   Most uncertain indices (first 20): {uncertainty_analysis['uncertain_indices'][:20]}")


        # === Attention Map Generation ===
    correct_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l][:num_attention_samples]
    wrong_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l][:num_attention_samples]

    attn_dir = os.path.join(output_dir, "attention_maps")
    os.makedirs(attn_dir, exist_ok=True)

    print("\n🧠 Generating attention maps for correct predictions...")
    for idx in correct_indices:
        input_tensor = all_inputs[idx].unsqueeze(0).to(device)
        label = all_labels[idx]
        if is_AST:
            attention_map.generate_enhanced_attention_maps(model, input_tensor, attn_dir, idx, label)
        else:
            attention_map.generate_enhanced_attention_maps_pretrained(model, input_tensor, attn_dir, idx, label)

    print("\n🧠 Generating attention maps for incorrect predictions...")
    for idx in wrong_indices:
        input_tensor = all_inputs[idx].unsqueeze(0).to(device)
        label = all_labels[idx]
        if is_AST:
            attention_map.generate_enhanced_attention_maps(model, input_tensor, attn_dir, idx, label)
        else:
            attention_map.generate_enhanced_attention_maps_pretrained(model, input_tensor, attn_dir, idx, label)


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
        "Balanced_Accuracy": comp_metrics['balanced_accuracy'],
        "Precision": comp_metrics['precision'],
        "Recall": comp_metrics['recall'],
        "F1_Score": comp_metrics['f1_score'],
        "Matthews_Corrcoef": comp_metrics['matthews_corrcoef'],
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "EER": eer,
        "EER_Threshold": threshold,
        
        # Additional metrics
        "ROC_AUC": comp_metrics['roc_auc'],
        "PR_AUC": comp_metrics['pr_auc'],
        "Optimal_Threshold": comp_metrics['optimal_threshold'],
        
        # Calibration metrics
        "Expected_Calibration_Error": calibration['expected_calibration_error'],
        
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
        
        # Class distribution metrics
        "Bonafide_Count": class_dist['bonafide_count'],
        "Spoof_Count": class_dist['spoof_count'],
        "Bonafide_Mean_Prob": class_dist['bonafide_prob_stats']['mean'],
        "Bonafide_Std_Prob": class_dist['bonafide_prob_stats']['std'],
        "Spoof_Mean_Prob": class_dist['spoof_prob_stats']['mean'],
        "Spoof_Std_Prob": class_dist['spoof_prob_stats']['std'],
        "Class_Separation": abs(class_dist['spoof_prob_stats']['mean'] - class_dist['bonafide_prob_stats']['mean'])
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
        "comprehensive_metrics": comp_metrics,
        "class_distribution": class_dist,
        "calibration_analysis": calibration,
        "threshold_analysis": threshold_analysis,
        "uncertainty_analysis": uncertainty_analysis,
        "wrong_analysis": wrong_analysis,
        "probability_stats": prob_stats,
        
        # Output directory
        "output_dir": output_dir if save_plots else None
    }
    
    print(f"\n✅ Enhanced benchmark completed! Results {'saved to ' + output_dir if save_plots else 'generated'}")
    
    return results