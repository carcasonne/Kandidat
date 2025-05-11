"""
Quick plotting script for creating custom plots on demand.
This script can be run from the command line to create custom plots without modifying the main code.

Examples:
    # Compare runs A and B with metrics X and Y:
    python quick_plot.py --runs AST_20K PRETRAINED_20K --metrics "Train Accuracy" "Val Accuracy" --title "AST vs Pretrained (20K)" --output "ast_vs_pretrained_20k.png"
    
    # Compare all AST runs with train accuracy:
    python quick_plot.py --run_group AST --metrics "Train Accuracy" --output "ast_train_accuracy.png"
    
    # Compare training and validation accuracy for a specific run:
    python quick_plot.py --runs AST_100K --metrics "Train Accuracy" "Val Accuracy" --output "ast_100k_train_vs_val.png"
"""

import argparse
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import sys

# Import our modules
from model_runs import (
    ModelRun, AST_RUNS, PRETRAINED_RUNS, ALL_RUNS, 
    AST_2K, AST_20K, AST_100K, PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K,
    get_runs_by_tag, get_runs_by_size, DataSize
)
from metric_categories import get_metric, MetricCategory
from visualizer import Visualizer
from wandb_login import login

def get_run_by_name(name):
    """Get a run by its variable name in model_runs.py"""
    # Try to get the run from model_runs module
    runs_module = sys.modules['model_runs']
    if hasattr(runs_module, name):
        return getattr(runs_module, name)
    
    # If not found, try to match by display name
    for run in ALL_RUNS:
        if run.display_name.lower() == name.lower() or run.id == name:
            return run
    
    raise ValueError(f"Run '{name}' not found")

def get_runs_by_group(group_name):
    """Get runs by group name (AST, PRETRAINED, ALL, SMALL, MEDIUM, LARGE)"""
    group_name = group_name.upper()
    
    # Check predefined groups
    if group_name == 'AST':
        return AST_RUNS
    elif group_name == 'PRETRAINED':
        return PRETRAINED_RUNS
    elif group_name == 'ALL':
        return ALL_RUNS
    elif group_name == 'SMALL':
        return get_runs_by_size(DataSize.SMALL)
    elif group_name == 'MEDIUM':
        return get_runs_by_size(DataSize.MEDIUM)
    elif group_name == 'LARGE':
        return get_runs_by_size(DataSize.LARGE)
    else:
        # Try to interpret as a tag
        return get_runs_by_tag(group_name.lower())

def parse_args():
    parser = argparse.ArgumentParser(description='Create custom plots of W&B runs and metrics')
    
    # Run selection options (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--runs', nargs='+', help='Specific runs to include (by variable name in model_runs.py)')
    run_group.add_argument('--run_group', type=str, help='Group of runs to include (AST, PRETRAINED, ALL, SMALL, MEDIUM, LARGE)')
    
    # Metric selection
    parser.add_argument('--metrics', nargs='+', required=True, help='Metrics to plot')
    
    # Plot options
    parser.add_argument('--title', type=str, help='Plot title')
    parser.add_argument('--output', type=str, required=True, help='Output filename')
    parser.add_argument('--smoothing', type=int, default=1, help='Smoothing window size (1 = no smoothing)')
    parser.add_argument('--plot_type', type=str, default='line', choices=['line', 'area'], help='Plot type')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 6], help='Figure size (width height)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='quick_plots', help='Output directory')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Login to W&B
    login()
    
    # Create visualizer
    visualizer = Visualizer(output_dir=Path(args.output_dir))
    
    # Get runs
    if args.runs:
        runs = [get_run_by_name(run_name) for run_name in args.runs]
    else:
        runs = get_runs_by_group(args.run_group)
    
    print(f"Creating plot with {len(runs)} runs and {len(args.metrics)} metrics...")
    
    if len(runs) == 1 and len(args.metrics) > 1:
        # Single run, multiple metrics
        fig = visualizer.compare_metrics(
            run=runs[0],
            metrics=args.metrics,
            title=args.title,
            figsize=tuple(args.figsize),
            smoothing=args.smoothing
        )
    elif len(runs) > 1 and len(args.metrics) == 1:
        # Multiple runs, single metric
        fig = visualizer.compare_runs(
            runs=runs,
            metric=args.metrics[0],
            title=args.title,
            figsize=tuple(args.figsize),
            smoothing=args.smoothing
        )
    else:
        # Custom plot with specific runs and metrics
        if len(args.metrics) == 1:
            # Same metric for all runs
            runs_and_metrics = {run: [args.metrics[0]] for run in runs}
        else:
            # Different runs might use different metrics (e.g., AST vs Pretrained)
            runs_and_metrics = {}
            for run in runs:
                if "AST" in run.display_name:
                    # Use standard metrics for AST
                    metrics_to_use = [m for m in args.metrics if 'Accuracy' in m or 'Loss' in m]
                    runs_and_metrics[run] = metrics_to_use
                else:
                    # Handle pretrained models differently if needed
                    metrics_to_use = []
                    for m in args.metrics:
                        if m == 'Train Accuracy' and 'Accuracy' in run.get_run_data(run).columns:
                            metrics_to_use.append('Accuracy')
                        else:
                            metrics_to_use.append(m)
                    runs_and_metrics[run] = metrics_to_use
        
        fig = visualizer.plot_custom(
            runs_and_metrics=runs_and_metrics,
            title=args.title,
            figsize=tuple(args.figsize),
            smoothing=args.smoothing,
            plot_type=args.plot_type
        )
    
    # Save the plot
    output_path = visualizer.save_figure(fig, args.output)
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    main()