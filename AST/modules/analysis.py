"""
Analysis module for audio deepfake detection models.
Provides tools for analyzing model performance and misclassified samples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
from pathlib import Path
import modules.results_cache as results_cache


class ModelAnalyzer:
    """Analyzer for model predictions and misclassifications."""
    
    def __init__(self, model, device='cuda', is_ast=True, model_path=None):
        """
        Initialize the model analyzer.
        
        Args:
            model: The trained model to analyze
            device: Device to run inference on
            is_ast: Whether the model is an AST model or pretrained model
            model_path: Path to model for caching results
        """
        self.model = model
        self.device = device
        self.is_ast = is_ast
        self.model_path = model_path
        self.model.eval()
        
        # Storage for analysis results
        self.reset_results()
        
    def reset_results(self):
        """Reset all stored analysis results."""
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
        self.misclassified_samples = []
        
    def analyze_dataset(self, dataloader, dataset_name="Unknown", verbose=True):
        """
        Analyze a complete dataset and identify misclassified samples.
        
        Args:
            dataloader: DataLoader containing the dataset
            dataset_name: Name of the dataset for reporting
            verbose: Whether to print progress information
        
        Returns:
            dict: Analysis results summary
        """
        if verbose:
            print(f"Analyzing {dataset_name} dataset...")
        
        self.reset_results()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", disable=not verbose)):
                # Handle different batch formats
                inputs, labels = self._extract_batch_data(batch)
                
                # Get model predictions
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Calculate predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                
                # Store results
                self.predictions.extend(predicted.cpu().numpy())
                self.ground_truth.extend(labels.cpu().numpy())
                self.confidences.extend(probs.max(dim=1)[0].cpu().numpy())
                
                # Identify misclassified samples
                self._record_misclassifications(batch_idx, dataloader.batch_size, 
                                              predicted, labels, probs)
        
        # Convert to numpy arrays
        self.predictions = np.array(self.predictions)
        self.ground_truth = np.array(self.ground_truth)
        self.confidences = np.array(self.confidences)
        
        # Generate summary
        summary = self._generate_summary(dataset_name, verbose)
        return summary
    
    def _extract_batch_data(self, batch):
        """Extract inputs and labels from batch based on format."""
        if self.is_ast:
            inputs = batch['input_values'].to(self.device)
            labels = batch['labels'].to(self.device)
        else:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = torch.tensor(labels).to(self.device) if not isinstance(labels, torch.Tensor) else labels.to(self.device)
            else:
                inputs = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
        return inputs, labels
    
    def _record_misclassifications(self, batch_idx, batch_size, predicted, labels, probs):
        """Record information about misclassified samples."""
        mask = predicted != labels
        if mask.any():
            misclassified_indices = torch.where(mask)[0]
            for idx in misclassified_indices:
                global_idx = batch_idx * batch_size + idx
                self.misclassified_samples.append({
                    'global_index': global_idx,
                    'batch_index': batch_idx,
                    'sample_index': idx.item(),
                    'predicted': predicted[idx].item(),
                    'actual': labels[idx].item(),
                    'confidence': probs[idx].max().item(),
                    'probs': probs[idx].cpu().numpy()
                })
    
    def _generate_summary(self, dataset_name, verbose):
        """Generate analysis summary."""
        summary = {
            'dataset_name': dataset_name,
            'total_samples': len(self.predictions),
            'total_misclassified': len(self.misclassified_samples),
            'accuracy': (self.predictions == self.ground_truth).mean(),
            'misclassification_rate': len(self.misclassified_samples) / len(self.predictions) if len(self.predictions) > 0 else 0,
        }
        
        if verbose:
            print(f"Analysis complete for {dataset_name}")
            print(f"Total samples: {summary['total_samples']}")
            print(f"Misclassified samples: {summary['total_misclassified']}")
            print(f"Accuracy: {summary['accuracy']:.4f}")
        
        return summary
    
    def get_misclassification_breakdown(self):
        """Get detailed breakdown of misclassifications."""
        if not self.misclassified_samples:
            return None
        
        # Analyze by class
        misclass_by_true_class = defaultdict(int)
        misclass_by_pred_class = defaultdict(int)
        confidence_by_error_type = defaultdict(list)
        
        for sample in self.misclassified_samples:
            true_class = sample['actual']
            pred_class = sample['predicted']
            confidence = sample['confidence']
            
            misclass_by_true_class[true_class] += 1
            misclass_by_pred_class[pred_class] += 1
            
            error_type = f"True_{true_class}_Pred_{pred_class}"
            confidence_by_error_type[error_type].append(confidence)
        
        return {
            'by_true_class': dict(misclass_by_true_class),
            'by_pred_class': dict(misclass_by_pred_class),
            'confidence_by_error_type': {k: np.array(v) for k, v in confidence_by_error_type.items()}
        }
    
    def get_low_confidence_samples(self, threshold=0.6):
        """Get samples with low confidence predictions."""
        low_conf_mask = self.confidences < threshold
        low_conf_samples = []
        
        for i, is_low_conf in enumerate(low_conf_mask):
            if is_low_conf:
                low_conf_samples.append({
                    'index': i,
                    'predicted': self.predictions[i],
                    'actual': self.ground_truth[i],
                    'confidence': self.confidences[i],
                    'correct': self.predictions[i] == self.ground_truth[i]
                })
        
        return low_conf_samples


class AnalysisVisualizer:
    """Handles visualization of analysis results."""
    
    @staticmethod
    def plot_confusion_matrix(ground_truth, predictions, class_names=None, save_path=None, title="Confusion Matrix"):
        """Plot confusion matrix."""
        if class_names is None:
            class_names = ['Real/Bonafide', 'Fake']
        
        cm = confusion_matrix(ground_truth, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_confidence_distribution(predictions, ground_truth, confidences, save_path=None, title="Confidence Distribution"):
        """Plot confidence distribution for correct vs incorrect predictions."""
        correct_mask = predictions == ground_truth
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'{title} - Histogram')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        data = [correct_conf, incorrect_conf]
        labels = ['Correct', 'Incorrect']
        plt.boxplot(data, labels=labels)
        plt.ylabel('Confidence')
        plt.title(f'{title} - Box Plot')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_class_performance_comparison(analyzer_results, save_path=None):
        """Compare performance across different classes/datasets."""
        if not isinstance(analyzer_results, list):
            analyzer_results = [analyzer_results]
        
        datasets = []
        accuracies = []
        
        for result in analyzer_results:
            if isinstance(result, dict) and 'dataset_name' in result:
                datasets.append(result['dataset_name'])
                accuracies.append(result['accuracy'])
        
        if not datasets:
            print("No valid results to plot")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, accuracies, color='skyblue', alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class AnalysisReporter:
    """Handles reporting and saving of analysis results."""
    
    @staticmethod
    def save_misclassified_samples(misclassified_samples, output_file):
        """Save misclassified samples information to CSV."""
        if not misclassified_samples:
            print("No misclassified samples to save.")
            return None
        
        data = []
        for sample in misclassified_samples:
            data.append({
                'global_index': sample['global_index'],
                'predicted_class': sample['predicted'],
                'actual_class': sample['actual'],
                'confidence': sample['confidence'],
                'real_prob': sample['probs'][0],
                'fake_prob': sample['probs'][1],
                'error_type': f"True_{sample['actual']}_Pred_{sample['predicted']}"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Misclassified samples info saved to {output_file}")
        return df
    
    @staticmethod
    def print_detailed_report(analyzer, dataset_name="Unknown"):
        """Print a detailed analysis report."""
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS REPORT FOR {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Basic metrics
        print(f"Total samples: {len(analyzer.predictions)}")
        accuracy = (analyzer.predictions == analyzer.ground_truth).mean()
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total misclassified: {len(analyzer.misclassified_samples)}")
        print(f"Misclassification rate: {len(analyzer.misclassified_samples) / len(analyzer.predictions):.4f}")
        
        # Misclassification breakdown
        breakdown = analyzer.get_misclassification_breakdown()
        if breakdown:
            print(f"\nMisclassifications by true class:")
            for class_id, count in breakdown['by_true_class'].items():
                class_name = 'Real/Bonafide' if class_id == 0 else 'Fake'
                print(f"  {class_name}: {count}")
            
            print(f"\nConfidence statistics for error types:")
            for error_type, confidences in breakdown['confidence_by_error_type'].items():
                print(f"  {error_type}: Mean={confidences.mean():.3f}, Std={confidences.std():.3f}")
        
        # Low confidence analysis
        low_conf_samples = analyzer.get_low_confidence_samples(threshold=0.7)
        print(f"\nLow confidence samples (< 0.7): {len(low_conf_samples)}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(analyzer.ground_truth, analyzer.predictions, 
                                 target_names=['Real/Bonafide', 'Fake']))


# Convenience functions for quick analysis
def quick_analysis(model, dataloader, dataset_name, device='cuda', is_ast=True, save_dir=None, model_path=None):
    """
    Perform a quick analysis of a model on a dataset.
    
    Args:
        model: The model to analyze
        dataloader: DataLoader for the dataset
        dataset_name: Name of the dataset
        device: Device to use
        is_ast: Whether model is AST format
        save_dir: Directory to save results (optional)
        model_path: Path to model for caching results
    
    Returns:
        ModelAnalyzer: The analyzer with results
    """
    analyzer = ModelAnalyzer(model, device=device, is_ast=is_ast, model_path=model_path)
    summary = analyzer.analyze_dataset(dataloader, dataset_name)
    
    # Print detailed report
    AnalysisReporter.print_detailed_report(analyzer, dataset_name)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Generate visualizations
    AnalysisVisualizer.plot_confusion_matrix(
        analyzer.ground_truth, analyzer.predictions,
        save_path=os.path.join(save_dir, f'{dataset_name}_confusion.png') if save_dir else None,
        title=f'{dataset_name} Confusion Matrix'
    )
    
    AnalysisVisualizer.plot_confidence_distribution(
        analyzer.predictions, analyzer.ground_truth, analyzer.confidences,
        save_path=os.path.join(save_dir, f'{dataset_name}_confidence.png') if save_dir else None,
        title=f'{dataset_name} Confidence Distribution'
    )
    
    # Save misclassified samples if save_dir provided
    if save_dir:
        AnalysisReporter.save_misclassified_samples(
            analyzer.misclassified_samples,
            os.path.join(save_dir, f'{dataset_name}_misclassified.csv')
        )
    
    return analyzer


def compare_models(models_and_loaders, save_dir=None):
    """
    Compare multiple models on their respective datasets.
    
    Args:
        models_and_loaders: List of tuples (model, dataloader, dataset_name, is_ast)
        save_dir: Directory to save comparison results
    
    Returns:
        List of analysis results
    """
    results = []
    analyzers = []
    
    for model, dataloader, dataset_name, is_ast in models_and_loaders:
        print(f"\nAnalyzing {dataset_name}...")
        analyzer = ModelAnalyzer(model, is_ast=is_ast)
        summary = analyzer.analyze_dataset(dataloader, dataset_name, verbose=False)
        
        results.append(summary)
        analyzers.append(analyzer)
        
        print(f"{dataset_name}: Accuracy = {summary['accuracy']:.4f}")
    
    # Create comparison visualization
    AnalysisVisualizer.plot_class_performance_comparison(
        results,
        save_path=os.path.join(save_dir, 'model_comparison.png') if save_dir else None
    )
    
    return results, analyzers