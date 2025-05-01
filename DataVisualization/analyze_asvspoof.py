#!/usr/bin/env python3
"""
ASVspoof Dataset Analysis Pipeline

This script performs comprehensive analysis of the ASVspoof 21 dataset,
generating statistics, visualizations, and publication-quality figures.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add the required directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our custom modules
from Visualizers.SpectrogramVisualizer import SpectrogramVisualizer
from Analyzers.DatasetAnalyzer import DatasetAnalyzer
from Visualizers.PaperFigureGenerator import PaperFigureGenerator
from Core.SpectrogramBuilder import SpectrogramBuilder

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ASVspoof Dataset Analysis Tool')
    
    parser.add_argument('--data_dir', type=str, default='spectrograms/ASVSpoof',
                        help='Directory containing the dataset (default: spectrograms/ASVSpoof)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs (default: output)')
    parser.add_argument('--max_files', type=int, default=100,
                        help='Maximum number of files to analyze per category (default: 100)')
    parser.add_argument('--prefix', type=str, default='asvspoof_',
                        help='Prefix for output filenames (default: asvspoof_)')
    parser.add_argument('--colormap', type=str, default='inferno',
                        help='Colormap to use for spectrogram visualization (default: inferno)')
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip dataset analysis and use existing statistics')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip generating paper figures')
    
    return parser.parse_args()

def main():
    """Main function for the ASVspoof dataset analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"\n===== ASVspoof Dataset Analysis Pipeline =====")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max files per category: {args.max_files}")
    
    # Create output directories
    analysis_dir = os.path.join(args.output_dir, "analysis")
    figures_dir = os.path.join(args.output_dir, "paper_figures")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Path to statistics file
    stats_file = os.path.join(analysis_dir, f"{args.prefix}statistics.json")
    
    # Step 1: Dataset Analysis
    if not args.skip_analysis:
        print("\n----- Step 1: Comprehensive Dataset Analysis -----")
        # Initialize analyzer
        analyzer = DatasetAnalyzer(
            data_dir=args.data_dir,
            output_dir=analysis_dir
        )
        
        # Run comprehensive analysis
        stats = analyzer.run_comprehensive_analysis(
            max_files=args.max_files,
            output_prefix=args.prefix
        )
    else:
        print("\n----- Step 1: Using existing statistics -----")
        print(f"Statistics file: {stats_file}")
    
    # Step 2: Generate Sample Visualizations
    print("\n----- Step 2: Generating Sample Visualizations -----")
    
    # Initialize builder and visualizer
    builder = SpectrogramBuilder(data_dir=args.data_dir)
    visualizer = SpectrogramVisualizer(output_dir=args.output_dir)
    
    # Load sample spectrograms
    real_specs, fake_specs = builder.get_sample_spectrograms(
        num_real=3,
        num_fake=3
    )
    
    print(f"Loaded {len(real_specs)} real and {len(fake_specs)} fake samples for visualization.")
    
    # Visualize real vs fake samples
    visualizer.visualize_real_vs_fake(
        real_specs, 
        fake_specs, 
        cmap=args.colormap,
        output_filename=f"{args.prefix}sample_comparison.png"
    )
    
    # Visualize differences for the first pair
    if real_specs and fake_specs:
        real_key = list(real_specs.keys())[0]
        fake_key = list(fake_specs.keys())[0]
        
        visualizer.visualize_difference(
            real_specs[real_key], 
            fake_specs[fake_key],
            title=f"Difference between {real_key} and {fake_key}",
            output_filename=f"{args.prefix}sample_difference.png"
        )
    
    # Step 3: Generate Publication Figures
    if not args.skip_figures:
        print("\n----- Step 3: Generating Publication-Quality Figures -----")
        
        # Initialize figure generator
        figure_generator = PaperFigureGenerator(
            data_dir=args.data_dir,
            output_dir=figures_dir
        )
        
        # Generate all figures
        figures = figure_generator.generate_all_figures(
            stats_file=stats_file,
            output_prefix="figure_"
        )
        
        print(f"Generated {len(figures)} publication-quality figures.")
    else:
        print("\n----- Step 3: Skipping Publication Figures -----")
    
    print(f"\n===== Analysis Complete! =====")
    print(f"All outputs saved to {args.output_dir}/")
    print(f"   - Analysis results: {analysis_dir}/")
    print(f"   - Sample visualizations: {args.output_dir}/")
    print(f"   - Publication figures: {figures_dir}/")
    print("\nThese visualizations can be directly used in your paper about the ASVspoof 21 dataset.")

if __name__ == "__main__":
    main()