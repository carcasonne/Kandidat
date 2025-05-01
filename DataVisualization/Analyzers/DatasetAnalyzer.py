import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec
import random

class DatasetAnalyzer:
    """
    A class for analyzing and visualizing characteristics of spectrogram datasets,
    particularly focused on ASVspoof datasets with bonafide and fake samples.
    """
    
    def __init__(self, data_dir="spectrograms/ASVSpoof", output_dir="output/analysis"):
        """
        Initialize the DatasetAnalyzer.
        
        Args:
            data_dir (str): Directory containing the dataset
            output_dir (str): Directory to save analysis outputs
        """
        self.data_dir = data_dir
        self.bonafide_dir = os.path.join(data_dir, "bonafide")
        self.fake_dir = os.path.join(data_dir, "fake")
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-quality figure settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.7,
            'figure.dpi': 300
        })
    
    def load_all_spectrograms(self, category="bonafide", max_files=None, random_sample=False):
        """
        Load all spectrograms from a specific category.
        
        Args:
            category (str): "bonafide" or "fake"
            max_files (int, optional): Maximum number of files to load
            random_sample (bool): Whether to take a random sample instead of first N
            
        Returns:
            dict: Dictionary with filenames as keys and spectrogram data as values
        """
        target_dir = self.bonafide_dir if category == "bonafide" else self.fake_dir
        
        if not os.path.exists(target_dir):
            print(f"Directory not found: {target_dir}")
            return {}
        
        spectrograms = {}
        files = [f for f in os.listdir(target_dir) if f.endswith('.npy')]
        
        if max_files is not None:
            if random_sample and len(files) > max_files:
                files = random.sample(files, max_files)
            else:
                files = files[:max_files]
        
        for filename in files:
            file_path = os.path.join(target_dir, filename)
            try:
                spec_data = np.load(file_path)
                spectrograms[filename] = spec_data
            except Exception as e:
                print(f"Error loading spectrogram from {file_path}: {e}")
        
        return spectrograms
    
    def get_dataset_statistics(self, max_files=100, random_sample=True):
        """
        Calculate comprehensive statistics about the dataset.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            random_sample (bool): Whether to take a random sample
            
        Returns:
            dict: Dictionary containing dataset statistics
        """
        bonafide_specs = self.load_all_spectrograms("bonafide", max_files, random_sample)
        fake_specs = self.load_all_spectrograms("fake", max_files, random_sample)
        
        # Count total files in each directory
        total_bonafide = len([f for f in os.listdir(self.bonafide_dir) 
                              if f.endswith('.npy')]) if os.path.exists(self.bonafide_dir) else 0
        total_fake = len([f for f in os.listdir(self.fake_dir) 
                          if f.endswith('.npy')]) if os.path.exists(self.fake_dir) else 0
        
        # Basic shape statistics
        bonafide_shapes = [spec.shape for spec in bonafide_specs.values()]
        fake_shapes = [spec.shape for spec in fake_specs.values()]
        
        # Most common shape
        most_common_shape_bonafide = Counter(bonafide_shapes).most_common(1)[0][0] if bonafide_shapes else None
        most_common_shape_fake = Counter(fake_shapes).most_common(1)[0][0] if fake_shapes else None
        
        # Calculate statistics for values
        bonafide_values = np.concatenate([spec.flatten() for spec in bonafide_specs.values()]) if bonafide_specs else np.array([])
        fake_values = np.concatenate([spec.flatten() for spec in fake_specs.values()]) if fake_specs else np.array([])
        
        # Per-spectrogram statistics
        bonafide_means = [np.mean(spec) for spec in bonafide_specs.values()]
        bonafide_stds = [np.std(spec) for spec in bonafide_specs.values()]
        bonafide_maxes = [np.max(spec) for spec in bonafide_specs.values()]
        bonafide_mins = [np.min(spec) for spec in bonafide_specs.values()]
        
        fake_means = [np.mean(spec) for spec in fake_specs.values()]
        fake_stds = [np.std(spec) for spec in fake_specs.values()]
        fake_maxes = [np.max(spec) for spec in fake_specs.values()]
        fake_mins = [np.min(spec) for spec in fake_specs.values()]
        
        return {
            "dataset_balance": {
                "total_bonafide": total_bonafide,
                "total_fake": total_fake,
                "ratio_bonafide_to_fake": total_bonafide / total_fake if total_fake > 0 else float('inf')
            },
            "shapes": {
                "bonafide_common_shape": most_common_shape_bonafide,
                "fake_common_shape": most_common_shape_fake,
                "bonafide_shape_counts": dict(Counter(bonafide_shapes).most_common()),
                "fake_shape_counts": dict(Counter(fake_shapes).most_common())
            },
            "overall_statistics": {
                "bonafide": {
                    "min": np.min(bonafide_values) if len(bonafide_values) > 0 else None,
                    "max": np.max(bonafide_values) if len(bonafide_values) > 0 else None,
                    "mean": np.mean(bonafide_values) if len(bonafide_values) > 0 else None,
                    "median": np.median(bonafide_values) if len(bonafide_values) > 0 else None,
                    "std": np.std(bonafide_values) if len(bonafide_values) > 0 else None
                },
                "fake": {
                    "min": np.min(fake_values) if len(fake_values) > 0 else None,
                    "max": np.max(fake_values) if len(fake_values) > 0 else None,
                    "mean": np.mean(fake_values) if len(fake_values) > 0 else None,
                    "median": np.median(fake_values) if len(fake_values) > 0 else None,
                    "std": np.std(fake_values) if len(fake_values) > 0 else None
                }
            },
            "per_spectrogram_statistics": {
                "bonafide": {
                    "mean_of_means": np.mean(bonafide_means) if bonafide_means else None,
                    "std_of_means": np.std(bonafide_means) if bonafide_means else None,
                    "mean_of_stds": np.mean(bonafide_stds) if bonafide_stds else None,
                    "mean_of_maxes": np.mean(bonafide_maxes) if bonafide_maxes else None,
                    "mean_of_mins": np.mean(bonafide_mins) if bonafide_mins else None
                },
                "fake": {
                    "mean_of_means": np.mean(fake_means) if fake_means else None,
                    "std_of_means": np.std(fake_means) if fake_means else None, 
                    "mean_of_stds": np.mean(fake_stds) if fake_stds else None,
                    "mean_of_maxes": np.mean(fake_maxes) if fake_maxes else None,
                    "mean_of_mins": np.mean(fake_mins) if fake_mins else None
                }
            },
            "sample_counts": {
                "bonafide_samples_analyzed": len(bonafide_specs),
                "fake_samples_analyzed": len(fake_specs)
            }
        }
    
    def visualize_dataset_balance(self, stats=None, output_filename="dataset_balance.png"):
        """
        Visualize the balance between bonafide and fake samples in the dataset.
        
        Args:
            stats (dict, optional): Statistics from get_dataset_statistics()
            output_filename (str): Output filename for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        if stats is None:
            stats = self.get_dataset_statistics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Bar plot
        categories = ['Bonafide', 'Fake']
        counts = [stats["dataset_balance"]["total_bonafide"], 
                 stats["dataset_balance"]["total_fake"]]
        
        bars = ax1.bar(categories, counts, color=['#2C7BB6', '#D7191C'])
        ax1.set_title('Dataset Composition', fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
               colors=['#2C7BB6', '#D7191C'], startangle=90)
        ax2.set_title('Proportion of Samples', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Balance visualization saved to {output_path}")
        
        return fig
    
    def visualize_spectrogram_distribution(self, max_files=100, output_filename="spectrogram_distribution.png"):
        """
        Visualize the distribution of values in spectrograms.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            output_filename (str): Output filename for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        bonafide_specs = self.load_all_spectrograms("bonafide", max_files, random_sample=True)
        fake_specs = self.load_all_spectrograms("fake", max_files, random_sample=True)
        
        # Extract means and standard deviations
        bonafide_means = [np.mean(spec) for spec in bonafide_specs.values()]
        bonafide_stds = [np.std(spec) for spec in bonafide_specs.values()]
        
        fake_means = [np.mean(spec) for spec in fake_specs.values()]
        fake_stds = [np.std(spec) for spec in fake_specs.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribution of means
        sns.histplot(bonafide_means, color='#2C7BB6', label='Bonafide', 
                    kde=True, ax=axes[0, 0], alpha=0.6)
        sns.histplot(fake_means, color='#D7191C', label='Fake', 
                    kde=True, ax=axes[0, 0], alpha=0.6)
        axes[0, 0].set_title('Distribution of Mean Values', fontweight='bold')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Distribution of standard deviations
        sns.histplot(bonafide_stds, color='#2C7BB6', label='Bonafide', 
                    kde=True, ax=axes[0, 1], alpha=0.6)
        sns.histplot(fake_stds, color='#D7191C', label='Fake', 
                    kde=True, ax=axes[0, 1], alpha=0.6)
        axes[0, 1].set_title('Distribution of Standard Deviations', fontweight='bold')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Scatter plot of mean vs std
        axes[1, 0].scatter(bonafide_means, bonafide_stds, color='#2C7BB6', 
                         alpha=0.6, label='Bonafide')
        axes[1, 0].scatter(fake_means, fake_stds, color='#D7191C', 
                         alpha=0.6, label='Fake')
        axes[1, 0].set_title('Mean vs Standard Deviation', fontweight='bold')
        axes[1, 0].set_xlabel('Mean Value')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].legend()
        
        # Box plots
        data = []
        categories = []
        
        for mean in bonafide_means:
            data.append(mean)
            categories.append('Bonafide (Mean)')
        
        for mean in fake_means:
            data.append(mean)
            categories.append('Fake (Mean)')
        
        for std in bonafide_stds:
            data.append(std)
            categories.append('Bonafide (Std)')
        
        for std in fake_stds:
            data.append(std)
            categories.append('Fake (Std)')
        
        df = pd.DataFrame({'Value': data, 'Category': categories})
        sns.boxplot(x='Category', y='Value', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('Box Plots of Mean and Std Values', fontweight='bold')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution visualization saved to {output_path}")
        
        return fig
    
    def visualize_energy_distribution(self, max_files=50, output_filename="energy_distribution.png"):
        """
        Visualize the energy distribution in spectrograms.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            output_filename (str): Output filename for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        bonafide_specs = self.load_all_spectrograms("bonafide", max_files, random_sample=True)
        fake_specs = self.load_all_spectrograms("fake", max_files, random_sample=True)
        
        # Calculate average energy distribution across frequency bins
        def get_freq_energy(specs):
            if not specs:
                return []
            
            # Get the most common shape to standardize
            shapes = [spec.shape for spec in specs.values()]
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            
            # Filter to only spectrograms with the most common shape
            filtered_specs = {k: v for k, v in specs.items() if v.shape == most_common_shape}
            
            if not filtered_specs:
                return []
            
            # Calculate mean energy across frequency bins (averaging across time)
            freq_energies = []
            for spec in filtered_specs.values():
                energy_by_freq = np.mean(spec, axis=1)  # Average across time dimension
                freq_energies.append(energy_by_freq)
            
            # Calculate average energy profile
            avg_energy_profile = np.mean(freq_energies, axis=0)
            return avg_energy_profile
        
        # Calculate average energy distribution across time frames
        def get_time_energy(specs):
            if not specs:
                return []
            
            # Get the most common shape to standardize
            shapes = [spec.shape for spec in specs.values()]
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            
            # Filter to only spectrograms with the most common shape
            filtered_specs = {k: v for k, v in specs.items() if v.shape == most_common_shape}
            
            if not filtered_specs:
                return []
            
            # Calculate mean energy across time frames (averaging across frequencies)
            time_energies = []
            for spec in filtered_specs.values():
                energy_by_time = np.mean(spec, axis=0)  # Average across frequency dimension
                time_energies.append(energy_by_time)
            
            # Calculate average energy profile
            avg_energy_profile = np.mean(time_energies, axis=0)
            return avg_energy_profile
        
        bonafide_freq_energy = get_freq_energy(bonafide_specs)
        fake_freq_energy = get_freq_energy(fake_specs)
        
        bonafide_time_energy = get_time_energy(bonafide_specs)
        fake_time_energy = get_time_energy(fake_specs)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot frequency energy distributions
        if len(bonafide_freq_energy) > 0 and len(fake_freq_energy) > 0:
            # Check if lengths match, if not, use the shorter one
            min_len = min(len(bonafide_freq_energy), len(fake_freq_energy))
            bonafide_freq_energy = bonafide_freq_energy[:min_len]
            fake_freq_energy = fake_freq_energy[:min_len]
            
            x = np.arange(len(bonafide_freq_energy))
            ax1.plot(x, bonafide_freq_energy, color='#2C7BB6', label='Bonafide')
            ax1.plot(x, fake_freq_energy, color='#D7191C', label='Fake')
            ax1.fill_between(x, bonafide_freq_energy, alpha=0.3, color='#2C7BB6')
            ax1.fill_between(x, fake_freq_energy, alpha=0.3, color='#D7191C')
            ax1.set_title('Average Energy Distribution Across Frequency Bins', fontweight='bold')
            ax1.set_xlabel('Frequency Bin')
            ax1.set_ylabel('Average Energy')
            ax1.legend()
            
            # Add inset with difference
            if len(bonafide_freq_energy) == len(fake_freq_energy):
                ax1_inset = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
                diff = bonafide_freq_energy - fake_freq_energy
                ax1_inset.plot(x, diff, color='black')
                ax1_inset.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax1_inset.set_title('Difference (Bonafide - Fake)')
                ax1_inset.set_xlabel('Frequency Bin')
                ax1_inset.set_ylabel('Difference')
        
        # Plot time energy distributions
        if len(bonafide_time_energy) > 0 and len(fake_time_energy) > 0:
            # Check if lengths match, if not, use the shorter one
            min_len = min(len(bonafide_time_energy), len(fake_time_energy))
            bonafide_time_energy = bonafide_time_energy[:min_len]
            fake_time_energy = fake_time_energy[:min_len]
            
            x = np.arange(len(bonafide_time_energy))
            ax2.plot(x, bonafide_time_energy, color='#2C7BB6', label='Bonafide')
            ax2.plot(x, fake_time_energy, color='#D7191C', label='Fake')
            ax2.fill_between(x, bonafide_time_energy, alpha=0.3, color='#2C7BB6')
            ax2.fill_between(x, fake_time_energy, alpha=0.3, color='#D7191C')
            ax2.set_title('Average Energy Distribution Across Time', fontweight='bold')
            ax2.set_xlabel('Time Frame')
            ax2.set_ylabel('Average Energy')
            ax2.legend()
            
            # Add inset with difference
            if len(bonafide_time_energy) == len(fake_time_energy):
                ax2_inset = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
                diff = bonafide_time_energy - fake_time_energy
                ax2_inset.plot(x, diff, color='black')
                ax2_inset.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax2_inset.set_title('Difference (Bonafide - Fake)')
                ax2_inset.set_xlabel('Time Frame')
                ax2_inset.set_ylabel('Difference')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Energy distribution visualization saved to {output_path}")
        
        return fig
    
    def visualize_statistical_significance(self, max_files=100, output_filename="statistical_significance.png"):
        """
        Visualize the statistical significance of differences between bonafide and fake spectrograms.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            output_filename (str): Output filename for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        bonafide_specs = self.load_all_spectrograms("bonafide", max_files, random_sample=True)
        fake_specs = self.load_all_spectrograms("fake", max_files, random_sample=True)
        
        # Extract features
        bonafide_means = [np.mean(spec) for spec in bonafide_specs.values()]
        bonafide_stds = [np.std(spec) for spec in bonafide_specs.values()]
        bonafide_maxes = [np.max(spec) for spec in bonafide_specs.values()]
        bonafide_mins = [np.min(spec) for spec in bonafide_specs.values()]
        bonafide_medians = [np.median(spec) for spec in bonafide_specs.values()]
        bonafide_skewness = [stats.skew(spec.flatten()) for spec in bonafide_specs.values()]
        bonafide_kurtosis = [stats.kurtosis(spec.flatten()) for spec in bonafide_specs.values()]
        
        fake_means = [np.mean(spec) for spec in fake_specs.values()]
        fake_stds = [np.std(spec) for spec in fake_specs.values()]
        fake_maxes = [np.max(spec) for spec in fake_specs.values()]
        fake_mins = [np.min(spec) for spec in fake_specs.values()]
        fake_medians = [np.median(spec) for spec in fake_specs.values()]
        fake_skewness = [stats.skew(spec.flatten()) for spec in fake_specs.values()]
        fake_kurtosis = [stats.kurtosis(spec.flatten()) for spec in fake_specs.values()]
        
        # Perform t-tests
        tests = {
            'Mean': stats.ttest_ind(bonafide_means, fake_means, equal_var=False),
            'Std Dev': stats.ttest_ind(bonafide_stds, fake_stds, equal_var=False),
            'Maximum': stats.ttest_ind(bonafide_maxes, fake_maxes, equal_var=False),
            'Minimum': stats.ttest_ind(bonafide_mins, fake_mins, equal_var=False),
            'Median': stats.ttest_ind(bonafide_medians, fake_medians, equal_var=False),
            'Skewness': stats.ttest_ind(bonafide_skewness, fake_skewness, equal_var=False),
            'Kurtosis': stats.ttest_ind(bonafide_kurtosis, fake_kurtosis, equal_var=False)
        }
        
        # Create a figure for the visualization
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Feature comparison
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Prepare data for boxplot
        feature_data = {
            'Mean (Bonafide)': bonafide_means,
            'Mean (Fake)': fake_means,
            'Std Dev (Bonafide)': bonafide_stds,
            'Std Dev (Fake)': fake_stds,
            'Skewness (Bonafide)': bonafide_skewness,
            'Skewness (Fake)': fake_skewness,
            'Kurtosis (Bonafide)': bonafide_kurtosis,
            'Kurtosis (Fake)': fake_kurtosis
        }
        
        df = pd.DataFrame({k: pd.Series(v) for k, v in feature_data.items()})
        sns.boxplot(data=df, ax=ax1)
        ax1.set_title('Feature Distribution Comparison', fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel('Value')
        
        # P-value visualization
        ax2 = fig.add_subplot(gs[0, 1])
        features = list(tests.keys())
        p_values = [test.pvalue for test in tests.values()]
        
        # Add labels and color based on significance
        colors = ['#D7191C' if p < 0.01 else '#FDAE61' if p < 0.05 else '#2C7BB6' for p in p_values]
        significance = ['p < 0.01 ***' if p < 0.01 else 'p < 0.05 **' if p < 0.05 else 'p ≥ 0.05' for p in p_values]
        
        # Create horizontal bar chart of -log10(p)
        log_p_values = [-np.log10(p) for p in p_values]
        bars = ax2.barh(features, log_p_values, color=colors)
        
        # Add p-value labels
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'p = {p:.4f}', va='center')
        
        ax2.set_title('Statistical Significance (-log10 p-value)', fontweight='bold')
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_ylabel('Feature')
        
        # Add a line for p=0.05 significance threshold
        ax2.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.7)
        ax2.text(-np.log10(0.05) + 0.1, len(features) - 0.5, 'p = 0.05', va='center')
        
        # Effect size visualization
        ax3 = fig.add_subplot(gs[1, :])
        
        # Calculate Cohen's d effect size
        effect_sizes = []
        for feature in features:
            if feature == 'Mean':
                d = (np.mean(bonafide_means) - np.mean(fake_means)) / np.sqrt(
                    (np.var(bonafide_means, ddof=1) + np.var(fake_means, ddof=1)) / 2)
            elif feature == 'Std Dev':
                d = (np.mean(bonafide_stds) - np.mean(fake_stds)) / np.sqrt(
                    (np.var(bonafide_stds, ddof=1) + np.var(fake_stds, ddof=1)) / 2)
            elif feature == 'Maximum':
                d = (np.mean(bonafide_maxes) - np.mean(fake_maxes)) / np.sqrt(
                    (np.var(bonafide_maxes, ddof=1) + np.var(fake_maxes, ddof=1)) / 2)
            elif feature == 'Minimum':
                d = (np.mean(bonafide_mins) - np.mean(fake_mins)) / np.sqrt(
                    (np.var(bonafide_mins, ddof=1) + np.var(fake_mins, ddof=1)) / 2)
            elif feature == 'Median':
                d = (np.mean(bonafide_medians) - np.mean(fake_medians)) / np.sqrt(
                    (np.var(bonafide_medians, ddof=1) + np.var(fake_medians, ddof=1)) / 2)
            elif feature == 'Skewness':
                d = (np.mean(bonafide_skewness) - np.mean(fake_skewness)) / np.sqrt(
                    (np.var(bonafide_skewness, ddof=1) + np.var(fake_skewness, ddof=1)) / 2)
            elif feature == 'Kurtosis':
                d = (np.mean(bonafide_kurtosis) - np.mean(fake_kurtosis)) / np.sqrt(
                    (np.var(bonafide_kurtosis, ddof=1) + np.var(fake_kurtosis, ddof=1)) / 2)
            effect_sizes.append(abs(d))
        
        # Define effect size categories
        effect_size_labels = []
        for d in effect_sizes:
            if d < 0.2:
                effect_size_labels.append('Negligible')
            elif d < 0.5:
                effect_size_labels.append('Small')
            elif d < 0.8:
                effect_size_labels.append('Medium')
            else:
                effect_size_labels.append('Large')
        
        # Create a bar chart of effect sizes
        colors = ['#FFFFBF' if label == 'Negligible' else
                 '#A6D96A' if label == 'Small' else
                 '#FDAE61' if label == 'Medium' else
                 '#D7191C' for label in effect_size_labels]
        
        bars = ax3.barh(features, effect_sizes, color=colors)
        
        # Add effect size labels
        for i, (bar, d, label) in enumerate(zip(bars, effect_sizes, effect_size_labels)):
            width = bar.get_width()
            ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'd = {d:.2f} ({label})', va='center')
        
        ax3.set_title('Effect Size (Cohen\'s d)', fontweight='bold')
        ax3.set_xlabel('Absolute Effect Size')
        ax3.set_ylabel('Feature')
        
        # Add effect size interpretation guide
        effect_size_guide = {'Negligible (d < 0.2)': '#FFFFBF',
                           'Small (0.2 ≤ d < 0.5)': '#A6D96A',
                           'Medium (0.5 ≤ d < 0.8)': '#FDAE61',
                           'Large (d ≥ 0.8)': '#D7191C'}
        
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in effect_size_guide.values()]
        ax3.legend(handles, effect_size_guide.keys(), 
                 loc='lower right', title='Effect Size Interpretation')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Statistical significance visualization saved to {output_path}")
        
        return fig
    
    def visualize_frequency_patterns(self, max_files=20, n_samples=3, output_filename="frequency_patterns.png"):
        """
        Visualize the frequency patterns in bonafide vs fake spectrograms.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            n_samples (int): Number of individual samples to show
            output_filename (str): Output filename for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        bonafide_specs = self.load_all_spectrograms("bonafide", max_files, random_sample=True)
        fake_specs = self.load_all_spectrograms("fake", max_files, random_sample=True)
        
        # Get the most common shape
        all_shapes = [spec.shape for spec in list(bonafide_specs.values()) + list(fake_specs.values())]
        most_common_shape = Counter(all_shapes).most_common(1)[0][0]
        
        # Filter to only spectrograms with the most common shape
        bonafide_filtered = {k: v for k, v in bonafide_specs.items() if v.shape == most_common_shape}
        fake_filtered = {k: v for k, v in fake_specs.items() if v.shape == most_common_shape}
        
        # Get sample spectrogram names to visualize
        bonafide_samples = list(bonafide_filtered.keys())[:n_samples]
        fake_samples = list(fake_filtered.keys())[:n_samples]
        
        # Get average spectrograms
        bonafide_avg = np.mean([spec for spec in bonafide_filtered.values()], axis=0) if bonafide_filtered else None
        fake_avg = np.mean([spec for spec in fake_filtered.values()], axis=0) if fake_filtered else None
        
        # Create figure
        n_rows = n_samples + 1  # +1 for average
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
        
        # Plot individual samples
        for i in range(n_samples):
            if i < len(bonafide_samples):
                bonafide_key = bonafide_samples[i]
                bonafide_spec = bonafide_filtered[bonafide_key]
                im = axes[i, 0].imshow(bonafide_spec, aspect='auto', origin='lower', cmap='inferno')
                axes[i, 0].set_title(f'Bonafide Sample: {bonafide_key}')
                plt.colorbar(im, ax=axes[i, 0])
            
            if i < len(fake_samples):
                fake_key = fake_samples[i]
                fake_spec = fake_filtered[fake_key]
                im = axes[i, 1].imshow(fake_spec, aspect='auto', origin='lower', cmap='inferno')
                axes[i, 1].set_title(f'Fake Sample: {fake_key}')
                plt.colorbar(im, ax=axes[i, 1])
        
        # Plot average spectrograms
        if bonafide_avg is not None:
            im = axes[n_samples, 0].imshow(bonafide_avg, aspect='auto', origin='lower', cmap='inferno')
            axes[n_samples, 0].set_title(f'Average of {len(bonafide_filtered)} Bonafide Spectrograms')
            plt.colorbar(im, ax=axes[n_samples, 0])
        
        if fake_avg is not None:
            im = axes[n_samples, 1].imshow(fake_avg, aspect='auto', origin='lower', cmap='inferno')
            axes[n_samples, 1].set_title(f'Average of {len(fake_filtered)} Fake Spectrograms')
            plt.colorbar(im, ax=axes[n_samples, 1])
        
        # Set common labels
        for ax in axes.flatten():
            ax.set_xlabel('Time Frame')
            ax.set_ylabel('Frequency Bin')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Frequency patterns visualization saved to {output_path}")
        
        return fig
    
    def run_comprehensive_analysis(self, max_files=100, output_prefix="asvspoof_"):
        """
        Run a comprehensive analysis of the dataset and save all visualizations.
        
        Args:
            max_files (int): Maximum number of files to analyze per category
            output_prefix (str): Prefix for output filenames
            
        Returns:
            dict: Dictionary with statistics about the dataset
        """
        print("Starting comprehensive analysis of ASVspoof dataset...")
        
        # Get dataset statistics
        print("Calculating dataset statistics...")
        stats = self.get_dataset_statistics(max_files)
        
        # Save statistics to JSON file
        import json
        with open(os.path.join(self.output_dir, f"{output_prefix}statistics.json"), 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            stats_serializable = {}
            for category, values in stats.items():
                if isinstance(values, dict):
                    stats_serializable[category] = {}
                    for key, value in values.items():
                        if isinstance(value, dict):
                            stats_serializable[category][key] = {}
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, np.ndarray):
                                    stats_serializable[category][key][subkey] = subvalue.tolist()
                                elif isinstance(subvalue, np.number):
                                    stats_serializable[category][key][subkey] = subvalue.item()
                                else:
                                    stats_serializable[category][key][subkey] = subvalue
                        elif isinstance(value, np.ndarray):
                            stats_serializable[category][key] = value.tolist()
                        elif isinstance(value, np.number):
                            stats_serializable[category][key] = value.item()
                        else:
                            stats_serializable[category][key] = value
                else:
                    stats_serializable[category] = values
            
            json.dump(stats_serializable, f, indent=4)
        
        print("Creating dataset balance visualization...")
        self.visualize_dataset_balance(stats, f"{output_prefix}dataset_balance.png")
        
        print("Creating spectrogram distribution visualization...")
        self.visualize_spectrogram_distribution(max_files, f"{output_prefix}spectrogram_distribution.png")
        
        print("Creating energy distribution visualization...")
        self.visualize_energy_distribution(max_files, f"{output_prefix}energy_distribution.png")
        
        print("Creating statistical significance visualization...")
        self.visualize_statistical_significance(max_files, f"{output_prefix}statistical_significance.png")
        
        print("Creating frequency patterns visualization...")
        self.visualize_frequency_patterns(20, 3, f"{output_prefix}frequency_patterns.png")
        
        print(f"Comprehensive analysis complete. All results saved to {self.output_dir}/")
        
        return stats