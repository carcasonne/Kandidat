import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from metric_categories import MetricCategory, MetricInfo, get_metric
from model_runs import ModelRun

def set_theme():
    """Set the default plot theme"""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "axes.facecolor": "white",      
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "serif",         
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "savefig.dpi": 300,             
        "savefig.transparent": False,
        "figure.dpi": 100
    })

def fetch_run_data(run: ModelRun) -> pd.DataFrame:
    """Fetch data for a specific run from W&B"""
    api = wandb.Api()
    wandb_run = api.run(run.full_path)
    
    # Retrieve full unsampled history
    history = wandb_run.scan_history()
    return pd.DataFrame(list(history))

def get_run_metric_data(run_data: pd.DataFrame, metric_name: str) -> pd.Series:
    """Extract a specific metric from run data"""
    if metric_name not in run_data.columns:
        return pd.Series(dtype=float)  # Return empty series if metric not found
    
    return run_data[metric_name].dropna().reset_index(drop=True)

class Visualizer:
    """Main visualization class for creating various plots"""
    
    def __init__(self, output_dir: Union[str, Path] = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_theme()
        self.run_data_cache = {}  # Cache for run data to avoid repeated API calls
    
    def get_run_data(self, run: ModelRun) -> pd.DataFrame:
        """Get data for a run, using cache if available"""
        if run.id not in self.run_data_cache:
            print(f"Fetching data for run {run.display_name}...")
            self.run_data_cache[run.id] = fetch_run_data(run)
        return self.run_data_cache[run.id]
    
    def plot_scalar_metrics(self, 
                           runs: List[ModelRun], 
                           metrics: List[str], 
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6),
                           smoothing: int = 1,
                           plot_type: str = 'line') -> plt.Figure:
        """
        Create a plot of scalar metrics across multiple runs
        
        Args:
            runs: List of ModelRun objects to include
            metrics: List of metric names to plot
            title: Optional title for the plot
            figsize: Figure size (width, height)
            smoothing: Window size for moving average smoothing (1 = no smoothing)
            plot_type: 'line' or 'area' for different plot styles
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a colormap for the runs
        n_runs = len(runs)
        cmap = cm.get_cmap('tab10')
        norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))
        
        # Plot each run/metric combination
        for i, run in enumerate(runs):
            run_data = self.get_run_data(run)
            run_color = cmap(norm(i))
            
            for metric_name in metrics:
                try:
                    metric_info = get_metric(metric_name)
                    metric_series = get_run_metric_data(run_data, metric_name)
                    
                    if len(metric_series) == 0:
                        print(f"Warning: No data found for metric '{metric_name}' in run '{run.display_name}'")
                        continue
                    
                    # Apply smoothing if requested
                    if smoothing > 1:
                        metric_series = metric_series.rolling(window=smoothing, min_periods=1).mean()
                    
                    # Get line style
                    line_style = metric_info.preferred_style or '-'
                    
                    # Choose color - use metric's preferred color if available, otherwise use run color
                    color = metric_info.preferred_color or run_color
                    
                    # Generate label
                    label = f"{run.display_name} - {metric_info.display_name}"
                    
                    # Plot based on type
                    if plot_type == 'line':
                        ax.plot(metric_series, label=label, color=color, linestyle=line_style)
                    elif plot_type == 'area':
                        ax.fill_between(range(len(metric_series)), 
                                       metric_series, 
                                       alpha=0.3, 
                                       color=color)
                        ax.plot(metric_series, label=label, color=color, linestyle=line_style)
                        
                except KeyError as e:
                    print(f"Error plotting {metric_name} for {run.display_name}: {e}")
        
        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            metric_names = [get_metric(m).display_name for m in metrics if m in get_metric(m).name]
            ax.set_title(f"{', '.join(metric_names)} - Comparison")
            
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        
        return fig
    

    def plot_confusion_matrix_between_runs(self,
                          run1: ModelRun,
                          run2: ModelRun,
                          metric_type: str,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6),
                          step: int = -1) -> plt.Figure:

        def load_and_plot(ax, run: ModelRun, conf_type: str):
            file_path = Path(f"wandb_data/{run.shortname}_conf_{conf_type}.csv")
            if not file_path.exists():
                raise FileNotFoundError(f"Confusion matrix file not found: {file_path}")

            df = pd.read_csv(file_path)
            conf_mat = df.pivot(index="Actual", columns="Predicted", values="nPredictions").fillna(0).astype(int)

            annotations = conf_mat.astype(str) + "\n" + (
                conf_mat.div(conf_mat.sum(axis=1), axis=0) * 100
            ).round(1).astype(str) + "%"
            sns.heatmap(conf_mat, annot=annotations, fmt='', cmap='Blues', ax=ax, cbar=False)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{run.display_name}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        load_and_plot(axes[0], run1, metric_type)
        load_and_plot(axes[1], run2, metric_type)
        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    def plot_confusion_matrix(self, 
                          run: ModelRun, 
                          metric_name: str = None,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6),
                          step: int = -1,
                          both: bool = False) -> plt.Figure:
        """
        Plot a confusion matrix from a CSV file for a specific run.

        Args:
            run: ModelRun object
            metric_name: One of 'train_conf_mat' or 'val_conf_mat'
            title: Optional title
            figsize: Size of the figure
            step: Unused, for compatibility

        Returns:
            Matplotlib Figure
        """
        # Determine file path
        def load_and_plot(ax, conf_type: str):
            file_path = Path(f"wandb_data/{run.shortname}_conf_{conf_type}.csv")
            if not file_path.exists():
                raise FileNotFoundError(f"Confusion matrix file not found: {file_path}")

            df = pd.read_csv(file_path)
            conf_mat = df.pivot(index="Actual", columns="Predicted", values="nPredictions").fillna(0).astype(int)

            annotations = conf_mat.astype(str) + "\n" + (
                conf_mat.div(conf_mat.sum(axis=1), axis=0) * 100
            ).round(1).astype(str) + "%"
            sns.heatmap(conf_mat, annot=annotations, fmt='', cmap='Blues', ax=ax, cbar=False)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{conf_type.capitalize()}")

        if both:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            load_and_plot(axes[0], "train")
            load_and_plot(axes[1], "val")
            if title:
                fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            conf_type = 'train' if 'train' in metric_name.lower() else 'val'
            fig, ax = plt.subplots(figsize=figsize)
            load_and_plot(ax, conf_type)
            if title:
                ax.set_title(title)
            fig.tight_layout()

        return fig
    
    def compare_runs(self, 
                    runs: List[ModelRun], 
                    metric: str,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    smoothing: int = 1) -> plt.Figure:
        """
        Compare a single metric across multiple runs
        
        Args:
            runs: List of ModelRun objects to include
            metric: Metric name to compare
            title: Optional title
            figsize: Figure size
            smoothing: Window size for moving average smoothing
            
        Returns:
            Matplotlib figure object
        """
        return self.plot_scalar_metrics(runs, [metric], title, figsize, smoothing)
    
    def compare_metrics(self, 
                       run: ModelRun, 
                       metrics: List[str],
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       smoothing: int = 1) -> plt.Figure:
        """
        Compare multiple metrics for a single run
        
        Args:
            run: ModelRun object
            metrics: List of metric names to compare
            title: Optional title
            figsize: Figure size
            smoothing: Window size for moving average smoothing
            
        Returns:
            Matplotlib figure object
        """
        return self.plot_scalar_metrics([run], metrics, title, figsize, smoothing)
    
    def plot_custom(self, 
                   runs_and_metrics: Dict[ModelRun, List[str]],
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 6),
                   smoothing: int = 1,
                   plot_type: str = 'line') -> plt.Figure:
        """
        Create a fully customized plot with specific runs and metrics
        
        Args:
            runs_and_metrics: Dictionary mapping runs to lists of metrics to plot
            title: Optional title
            figsize: Figure size
            smoothing: Window size for moving average smoothing
            plot_type: 'line' or 'area'
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create a colormap
        all_runs = list(runs_and_metrics.keys())
        n_runs = len(all_runs)
        cmap = cm.get_cmap('tab10')
        norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))

        # Plot each run/metric combination
        for i, (run, metrics) in enumerate(runs_and_metrics.items()):
            run_data = self.get_run_data(run)
            run_color = cmap(norm(i))

            for metric_name in metrics:
                try:
                    metric_info = get_metric(metric_name)
                    metric_series = get_run_metric_data(run_data, metric_name)

                    if len(metric_series) == 0:
                        print(f"Warning: No data found for metric '{metric_name}' in run '{run.display_name}'")
                        continue

                    # Apply smoothing
                    if smoothing > 1:
                        metric_series = metric_series.rolling(window=smoothing, min_periods=1).mean()

                    # Determine line style and color
                    line_style = metric_info.preferred_style or '-'
                    color = metric_info.preferred_color or run_color

                    # Generate label
                    label = f"{run.display_name} - {metric_info.display_name}"

                    # Plot
                    if plot_type == 'line':
                        ax.plot(metric_series, label=label, color=color, linestyle=line_style)
                    elif plot_type == 'area':
                        ax.fill_between(range(len(metric_series)),
                                       metric_series,
                                       alpha=0.3,
                                       color=color)
                        ax.plot(metric_series, label=label, color=color, linestyle=line_style)

                except KeyError as e:
                    print(f"Error plotting {metric_name} for {run.display_name}: {e}")

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Custom Metric Comparison")

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        return fig

    def compare_runs_with_distinct_colors(self,
                                    runs: List[ModelRun],
                                    metric: str,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 6),
                                    smoothing: int = 1,
                                    colormap: Optional[str] = None,
                                    run_colors: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Compare a single metric across multiple runs with distinct colors

        Args:
            runs: List of ModelRun objects to include
            metric: Metric name to compare
            title: Optional title
            figsize: Figure size
            smoothing: Window size for moving average smoothing
            colormap: Optional matplotlib colormap name to use if run_colors not provided
            run_colors: Optional dictionary mapping runs to specific colors

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use provided run_colors or generate from colormap
        if run_colors is None:
            # Create a colormap for the runs
            n_runs = len(runs)
            cmap = cm.get_cmap(colormap or 'tab10')
            norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))

            # Generate colors for each run
            generated_run_colors = {run: cmap(norm(i)) for i, run in enumerate(runs)}
        else:
            generated_run_colors = run_colors

        # Plot the metric for each run
        for run in runs:
            run_data = self.get_run_data(run)

            try:
                metric_info = get_metric(metric)
                metric_series = get_run_metric_data(run_data, metric)

                if len(metric_series) == 0:
                    print(f"Warning: No data found for metric '{metric}' in run '{run.display_name}'")
                    continue

                # Apply smoothing if requested
                if smoothing > 1:
                    metric_series = metric_series.rolling(window=smoothing, min_periods=1).mean()

                # Get line style from metric info
                line_style = metric_info.preferred_style or '-'

                # Get color from the generated/provided colors
                color = generated_run_colors.get(run.id)

                # Generate label (just the run name, since metric is the same)
                label = f"{run.display_name}"

                # Plot the line
                ax.plot(metric_series, label=label, color=color, linestyle=line_style)

            except KeyError as e:
                print(f"Error plotting {metric} for {run.display_name}: {e}")

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            metric_display = get_metric(metric).display_name
            ax.set_title(f"{metric_display} - Comparison")

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        return fig


    def compare_metrics_across_runs(self,
                            runs: List[ModelRun],
                            metrics: List[str],
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            smoothing: int = 1,
                            colormap: Optional[str] = 'tab10',
                            line_styles: Optional[Dict[str, str]] = None,
                            run_colors: Optional[Dict[str, str]] = None,
                            include_legend: bool = True) -> plt.Figure:
        """
        Compare multiple metrics across multiple runs in a single plot.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of metric names to compare
            title: Optional title for the plot
            figsize: Figure size (width, height)
            smoothing: Window size for moving average smoothing (1 = no smoothing)
            colormap: Matplotlib colormap name to use if run_colors not provided
            line_styles: Optional dictionary mapping metrics to specific line styles
            run_colors: Optional dictionary mapping run IDs to specific colors
            include_legend: Whether to include the legend (default: True)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Generate color map for runs if not provided
        if run_colors is None:
            n_runs = len(runs)
            cmap = cm.get_cmap(colormap)
            norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(runs)}

        # Default line styles for metrics if not provided
        if line_styles is None:
            line_styles = {}
            for i, metric_name in enumerate(metrics):
                try:
                    metric_info = get_metric(metric_name)
                    if metric_info.preferred_style:
                        line_styles[metric_name] = metric_info.preferred_style
                    # If no preferred style, alternate between solid and dashed
                    else:
                        line_styles[metric_name] = '-' if 'train' in metric_name.lower() else '--'
                except KeyError:
                    # Default to solid if metric not found
                    line_styles[metric_name] = '-'

        # Plot each run and metric combination
        for run in runs:
            run_data = self.get_run_data(run)
            run_color = run_colors.get(run.id)

            for metric_name in metrics:
                try:
                    metric_info = get_metric(metric_name)
                    metric_series = get_run_metric_data(run_data, metric_name)

                    if len(metric_series) == 0:
                        print(f"Warning: No data found for metric '{metric_name}' in run '{run.display_name}'")
                        continue

                    # Apply smoothing if requested
                    if smoothing > 1:
                        metric_series = metric_series.rolling(window=smoothing, min_periods=1).mean()

                    # Get line style from provided dictionary or default
                    line_style = line_styles.get(metric_name, '-.')

                    # Generate label combining run name and metric name
                    label = f"{run.display_name} - {metric_info.display_name}"

                    # Plot the line
                    ax.plot(metric_series, label=label, color=run_color, linestyle=line_style)

                except KeyError as e:
                    print(f"Error plotting {metric_name} for {run.display_name}: {e}")

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            metric_names = [get_metric(m).display_name for m in metrics if m in get_metric(m).name]
            ax.set_title(f"Comparison of {', '.join(metric_names)} Across Runs")

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

        if include_legend:
            ax.legend()

        fig.tight_layout()

        return fig



    def plot_final_metrics(self,
                        runs: List[ModelRun],
                        metrics: List[str],
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8),
                        run_colors: Optional[Dict[str, str]] = None,
                        metric_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> plt.Figure:
        """
        Create a bar plot comparing final metric values across runs.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of base metric names to compare (e.g., "Accuracy", "Precision")
            title: Optional title for the plot
            figsize: Figure size (width, height)
            run_colors: Optional dictionary mapping run IDs to specific colors
            metric_mapping: Optional dictionary mapping run type to metric name patterns
                            e.g., {"AST": {"Accuracy": "Train Accuracy", "Val Accuracy": "Val Accuracy"},
                                "Pretrained": {"Accuracy": "Accuracy", "Val Accuracy": "Val Accuracy"}}

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Number of runs and metrics
        n_runs = len(runs)
        n_metrics = len(metrics)

        # Generate colors for runs if not provided
        if run_colors is None:
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(runs)}

        # Width of bars and positions
        bar_width = 0.8 / n_metrics
        index = np.arange(n_runs)

        # For storing metric values to set y-axis limits
        all_values = []

        # Plot each metric as a group of bars
        for i, metric_name in enumerate(metrics):
            values = []

            for j, run in enumerate(runs):
                run_data = self.get_run_data(run)

                # Determine run type for metric mapping
                run_type = "AST" if "AST" in run.shortname else "Pretrained"


                if metric_mapping is None:
                    print("oh shit!")

                # Get the appropriate metric name for this run type
                if run_type in metric_mapping and metric_name in metric_mapping[run_type]:
                    actual_metric_name = metric_mapping[run_type][metric_name]
                else:
                    actual_metric_name = metric_name

                try:
                    # Try to get metric info - if it fails, we'll just use the raw name
                    try:
                        metric_info = get_metric(actual_metric_name)
                        display_name = metric_info.display_name
                    except:
                        display_name = actual_metric_name

                    metric_series = get_run_metric_data(run_data, actual_metric_name)

                    if len(metric_series) == 0:
                        print(f"Warning: No data found for metric '{actual_metric_name}' in run '{run.display_name}'")
                        values.append(0)
                        continue

                    # Get the final value
                    final_value = float(metric_series.iloc[-1])
                    values.append(final_value)
                    all_values.append(final_value)

                except KeyError as e:
                    print(f"Error getting {actual_metric_name} for {run.display_name}: {e}")
                    values.append(0)

            # Position for this metric's bars
            pos = index + i * bar_width - (n_metrics - 1) * bar_width / 2

            # Create a list to store the bars for this metric
            bars = []

            # Plot the bars for this metric with different colors by run
            for j, (pos_val, value, run) in enumerate(zip(pos, values, runs)):
                # Get color for this run
                color = run_colors.get(run.id)

                # Plot the bar with the run's color
                bar = ax.bar(pos_val, value, bar_width * 0.9,
                            color=color,
                            alpha=0.8)
                bars.append(bar[0])

                # Add value label on top of bar
                if value > 0:  # Only add label if value is positive
                    ax.text(pos_val, value + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

            # For legend - only add once per metric
            if i == 0:
                # Create custom legend for runs
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color=run_colors[run.id], lw=0,
                                        marker='s', markersize=10, label=run.display_name)
                                for run in runs]
                ax.legend(handles=legend_elements, loc='best')

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            metric_labels = []
            for m in metrics:
                try:
                    metric_labels.append(get_metric(m).display_name)
                except:
                    metric_labels.append(m)
            ax.set_title(f"Final Metrics Comparison")

        # Set x-axis ticks and labels
        ax.set_xticks(np.arange(len(metrics)))

        # Try to get display names for x-tick labels
        x_labels = []
        for m in metrics:
            try:
                x_labels.append(get_metric(m).display_name)
            except:
                x_labels.append(m)
        ax.set_xticklabels(x_labels)

        # Set y-axis limits with some headroom for labels
        if all_values:
            ax.set_ylim(0, max(all_values) * 1.15)

        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        fig.tight_layout()

        return fig


    def plot_final_metric_by_category(self,
                                runs: List[ModelRun],
                                metric: str,
                                group_by: Callable[[ModelRun], str],
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8),
                                color_by_group: bool = True) -> plt.Figure:
        """
        Create a grouped bar plot comparing a final metric value across runs,
        grouped by a category (e.g., data size, model type).

        Args:
            runs: List of ModelRun objects to include
            metric: Metric name to compare
            group_by: Function that takes a ModelRun and returns a group name
            title: Optional title for the plot
            figsize: Figure size (width, height)
            color_by_group: Whether to color bars by group (True) or by run (False)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Group runs
        grouped_runs = {}
        for run in runs:
            group = group_by(run)
            if group not in grouped_runs:
                grouped_runs[group] = []
            grouped_runs[group].append(run)

        # Number of groups and max runs per group
        n_groups = len(grouped_runs)
        max_runs_per_group = max(len(runs) for runs in grouped_runs.values())

        # Generate colors
        if color_by_group:
            # Color by group
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(n_groups-1, 1))
            group_colors = {group: cmap(norm(i)) for i, group in enumerate(grouped_runs.keys())}
        else:
            # Color by run
            all_runs = []
            for runs in grouped_runs.values():
                all_runs.extend(runs)
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(len(all_runs)-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(all_runs)}

        # Width of bars and positions
        group_width = 0.8
        bar_width = group_width / max_runs_per_group
        index = np.arange(n_groups)

        # For storing metric values to set y-axis limits
        all_values = []

        # Plot each group
        for i, (group, group_runs) in enumerate(grouped_runs.items()):
            for j, run in enumerate(group_runs):
                run_data = self.get_run_data(run)

                try:
                    metric_info = get_metric(metric)
                    metric_series = get_run_metric_data(run_data, metric)

                    if len(metric_series) == 0:
                        print(f"Warning: No data found for metric '{metric}' in run '{run.display_name}'")
                        continue

                    # Get the final value
                    final_value = float(metric_series.iloc[-1])
                    all_values.append(final_value)

                    # Position for this bar
                    pos = i + (j - len(group_runs)/2 + 0.5) * bar_width

                    # Plot the bar
                    if color_by_group:
                        color = group_colors[group]
                    else:
                        color = run_colors[run.id]

                    bar = ax.bar(pos, final_value, bar_width * 0.9,
                            color=color, alpha=0.8)

                    # Add value label on top of bar
                    height = bar[0].get_height()
                    ax.text(pos, height + 0.01,
                        f'{final_value:.3f}', ha='center', va='bottom', fontsize=9)

                    # Add run name below the bar for identification
                    ax.text(pos, -0.05, run.shortname, ha='center', va='top',
                            fontsize=8, rotation=45)

                except KeyError as e:
                    print(f"Error getting {metric} for {run.display_name}: {e}")

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            metric_display = get_metric(metric).display_name
            ax.set_title(f"Final {metric_display} Comparison by Group")

        # Set x-axis ticks and labels
        ax.set_xticks(index)
        ax.set_xticklabels(grouped_runs.keys())

        # Set y-axis limits with some headroom for labels
        if all_values:
            ax.set_ylim(0, max(all_values) * 1.15)

        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        # Add legend if coloring by run
        if not color_by_group:
            # Create custom legend handles
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=run_colors[run.id], lw=0,
                                    marker='s', markersize=10, label=run.display_name)
                            for run in all_runs]
            ax.legend(handles=legend_elements, loc='best')

        fig.tight_layout()

        return fig


    def plot_train_val_grouped_metrics(self,
                                runs: List[ModelRun],
                                metrics: List[str],
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 8),
                                run_colors: Optional[Dict[str, str]] = None,
                                metric_mapping: Optional[Dict[str, Dict[str, str]]] = None,
                                bar_width: float = 0.2,
                                space_between_groups: float = 1.0,
                                show_values: bool = True) -> plt.Figure:
        """
        Create a grouped bar chart displaying training and validation metrics side by side for each run.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of metrics to plot. Should contain pairs like ["Train Accuracy", "Val Accuracy"]
            title: Optional title for the plot
            figsize: Figure size (width, height)
            run_colors: Optional dictionary mapping run IDs or types to specific colors
            metric_mapping: Optional dictionary mapping run type to metric name patterns
            bar_width: Width of individual bars
            space_between_groups: Space between groups of bars for different runs
            show_values: Whether to show value labels on bars

        Returns:
            Matplotlib figure object
        """
        # Import Patch for legend
        from matplotlib.patches import Patch

        # Create figure with adjusted height to accommodate the legend at bottom
        fig, ax = plt.subplots(figsize=figsize)

        # Number of runs and metrics
        n_runs = len(runs)
        n_metric_pairs = len(metrics) // 2  # Assuming metrics come in train/val pairs

        # Adjust bar width based on number of metrics to prevent overcrowding
        if n_metric_pairs > 2:
            bar_width = min(bar_width, 0.15)

        # Total width for all bars in a group
        total_group_width = n_metric_pairs * 2 * bar_width + space_between_groups

        # Generate colors for runs if not provided
        if run_colors is None:
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(runs)}

        # Generate distinct colors for different metrics
        cyberpunk_colors = [
            "#00916e",  # Vibrant teal green
            "#d60036",  # Crimson red
            "#0057b7",  # Electric blue
            "#ff9500",  # Bright orange
            "#6a00ff",  # Vivid purple
            "#1a2e35",  # Dark midnight
            "#00b050",  # Bright green
            "#e53935",  # Bright red
            "#00a2ff",  # Bright blue
            "#ff6a00"   # Bright amber
        ]
        metric_colors = {i: cyberpunk_colors[i % len(cyberpunk_colors)] for i in range(n_metric_pairs)}

        # For storing metric values to set y-axis limits
        all_values = []

        # Create legend elements
        legend_elements = []

        # Process each run
        for i, run in enumerate(runs):
            run_data = self.get_run_data(run)

            # Determine run type for metric mapping
            run_type = "AST" if "AST" in run.shortname else "Pretrained"

            # Group start position
            group_start = i * total_group_width

            # Process each pair of metrics (train/val)
            for j in range(n_metric_pairs):
                train_idx = j * 2
                val_idx = j * 2 + 1

                if train_idx >= len(metrics) or val_idx >= len(metrics):
                    continue

                train_metric_name = metrics[train_idx]
                val_metric_name = metrics[val_idx]

                # Get metric short name for legend (without Train/Val prefix)
                metric_short_name = train_metric_name.replace("Train ", "")
                if " " not in metric_short_name:
                    # Handle case where it's just "Accuracy" without "Train" prefix
                    metric_short_name = train_metric_name

                # Get the appropriate metric names for this run type
                if metric_mapping and run_type in metric_mapping:
                    if train_metric_name in metric_mapping[run_type]:
                        actual_train_metric = metric_mapping[run_type][train_metric_name]
                    else:
                        actual_train_metric = train_metric_name

                    if val_metric_name in metric_mapping[run_type]:
                        actual_val_metric = metric_mapping[run_type][val_metric_name]
                    else:
                        actual_val_metric = val_metric_name
                else:
                    actual_train_metric = train_metric_name
                    actual_val_metric = val_metric_name

                # Get metric color
                metric_color = metric_colors[j]

                # Add to legend if first run (to avoid duplicates)
                if i == 0:
                    train_patch = Patch(facecolor=metric_color, alpha=0.9, edgecolor='black',
                                    label=f"{metric_short_name} (Train)")
                    val_patch = Patch(facecolor=metric_color, alpha=0.6, edgecolor='black',
                                    hatch='////', label=f"{metric_short_name} (Val)")
                    legend_elements.extend([train_patch, val_patch])

                # Process training metric
                try:
                    train_series = get_run_metric_data(run_data, actual_train_metric)
                    if len(train_series) > 0:
                        train_value = float(train_series.iloc[-1])
                        all_values.append(train_value)

                        # Bar position
                        train_pos = group_start + j * 2 * bar_width

                        # Plot bar with metric-specific color
                        train_bar = ax.bar(train_pos, train_value, bar_width * 0.9,
                                        color=metric_color, alpha=0.9,
                                        edgecolor='black', linewidth=0.5)

                        # Add value label if requested
                        if show_values:
                            # Calculate font size - smaller when there are more metrics
                            font_size = max(5, 8 - n_metric_pairs // 2)

                            # Only add labels to bars that are tall enough to be significant
                            if train_value > 0.05:  # Adjust threshold as needed
                                # Add value with rotation for better readability
                                ax.text(train_pos, train_value + 0.01,
                                    f'{train_value:.3f}',
                                    ha='center', va='bottom',
                                    fontsize=font_size,
                                    rotation=45 if n_metric_pairs > 2 else 0)
                except KeyError as e:
                    print(f"Error getting {actual_train_metric} for {run.display_name}: {e}")

                # Process validation metric
                try:
                    val_series = get_run_metric_data(run_data, actual_val_metric)
                    if len(val_series) > 0:
                        val_value = float(val_series.iloc[-1])
                        all_values.append(val_value)

                        # Bar position - right next to the training bar
                        val_pos = group_start + j * 2 * bar_width + bar_width

                        # Plot bar with same color but hatched
                        val_bar = ax.bar(val_pos, val_value, bar_width * 0.9,
                                    color=metric_color, alpha=0.6,
                                    edgecolor='black', linewidth=0.5, hatch='////')

                        # Add value label if requested
                        if show_values:
                            # Calculate font size - smaller when there are more metrics
                            font_size = max(5, 8 - n_metric_pairs // 2)

                            # Only add labels to bars that are tall enough to be significant
                            if val_value > 0.05:  # Adjust threshold as needed
                                # Add value with rotation for better readability
                                ax.text(val_pos, val_value + 0.01,
                                    f'{val_value:.3f}',
                                    ha='center', va='bottom',
                                    fontsize=font_size,
                                    rotation=45 if n_metric_pairs > 2 else 0)
                except KeyError as e:
                    print(f"Error getting {actual_val_metric} for {run.display_name}: {e}")

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Train vs Validation Metrics Comparison")

        # Set x-axis ticks at the middle of each group
        group_centers = []
        group_labels = []
        for i, run in enumerate(runs):
            group_center = i * total_group_width + (n_metric_pairs * bar_width)
            group_centers.append(group_center)
            group_labels.append(run.shortname)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')

        # Set y-axis limits with some headroom for labels
        if all_values:
            max_value = max(all_values)
            # Add more headroom if we're showing rotated value labels
            headroom = 1.25 if n_metric_pairs > 2 and show_values else 1.15
            ax.set_ylim(0, max_value * headroom)

        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        # Add legend below the plot
        # First adjust the subplot parameters to create space at the bottom
        bottom_space = 0.25 if n_metric_pairs > 2 else 0.2
        fig.subplots_adjust(bottom=bottom_space)

        # Then place the legend below the axes
        ax.legend(handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),  # Position below the axes
                ncol=min(len(legend_elements), 4),  # Arrange in multiple columns
                frameon=False)

        fig.tight_layout()

        return fig



    def plot_train_val_metrics_dot_plot(self,
                                runs: List[ModelRun],
                                metrics: List[str],
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 10),
                                run_colors: Optional[Dict[str, str]] = None,
                                metric_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> plt.Figure:
        """
        Create an improved dot plot showing training and validation metrics as connected dots.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of metrics to plot (should be pairs of train/val)
            title: Optional title for the plot
            figsize: Figure size (width, height)
            run_colors: Optional dictionary mapping run IDs or types to specific colors
            metric_mapping: Optional dictionary mapping run type to metric name patterns

        Returns:
            Matplotlib figure object
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        # Number of metric pairs
        n_metric_pairs = len(metrics) // 2

        # Create subplots - one per metric type
        fig, axes = plt.subplots(n_metric_pairs, 1, figsize=figsize, sharex=True)
        if n_metric_pairs == 1:
            axes = [axes]

        # Generate colors for runs if not provided
        if run_colors is None:
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(len(runs)-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(runs)}

        # For storing all values to set x-axis limits
        all_values = []
        run_positions = np.arange(len(runs)) * 2  # Doubled spacing for better readability

        # Process each metric pair in a separate subplot
        for ax_idx, j in enumerate(range(0, len(metrics), 2)):
            ax = axes[ax_idx]

            if j + 1 >= len(metrics):
                continue

            train_metric_name = metrics[j]
            val_metric_name = metrics[j+1]

            # Get metric short name for axis label
            metric_short_name = train_metric_name.replace("Train ", "")
            #if " " not in metric_short_name:
            #    metric_short_name = train_metric_name

            # Process each run
            for i, run in enumerate(runs):
                run_data = self.get_run_data(run)

                # Determine run type for metric mapping
                run_type = "AST" if "AST" in run.shortname else "Pretrained"

                # Get run color
                if run.id in run_colors:
                    run_color = run_colors[run.id]
                elif run_type in run_colors:
                    run_color = run_colors[run_type]
                else:
                    run_color = f"C{i}"

                # Get the appropriate metric names for this run type
                if metric_mapping and run_type in metric_mapping:
                    if train_metric_name in metric_mapping[run_type]:
                        actual_train_metric = metric_mapping[run_type][train_metric_name]
                    else:
                        actual_train_metric = train_metric_name

                    if val_metric_name in metric_mapping[run_type]:
                        actual_val_metric = metric_mapping[run_type][val_metric_name]
                    else:
                        actual_val_metric = val_metric_name
                else:
                    actual_train_metric = train_metric_name
                    actual_val_metric = val_metric_name

                # Get train value
                try:
                    train_series = get_run_metric_data(run_data, actual_train_metric)
                    train_value = float(train_series.iloc[-1]) if len(train_series) > 0 else np.nan
                    all_values.append(train_value)
                except KeyError as e:
                    print(f"Error getting {actual_train_metric} for {run.display_name}: {e}")
                    train_value = np.nan

                # Get val value
                try:
                    val_series = get_run_metric_data(run_data, actual_val_metric)
                    val_value = float(val_series.iloc[-1]) if len(val_series) > 0 else np.nan
                    all_values.append(val_value)
                except KeyError as e:
                    print(f"Error getting {actual_val_metric} for {run.display_name}: {e}")
                    val_value = np.nan

                # Plot the train and val values horizontally
                if not np.isnan(train_value) and not np.isnan(val_value):
                    # Horizontal line connecting train and val
                    ax.plot([train_value, val_value], [run_positions[i], run_positions[i]],
                        color=run_color, alpha=0.7, linestyle='-', linewidth=2)

                    # Train marker
                    ax.scatter(train_value, run_positions[i], s=100, color=run_color,
                            edgecolor='black', linewidth=1, zorder=10, marker='o')

                    # Val marker
                    ax.scatter(val_value, run_positions[i], s=100, color=run_color,
                            edgecolor='black', linewidth=1, zorder=10,
                            alpha=0.6, marker='s')

                    # Add value labels
                    ax.annotate(f'{train_value:.3f}', xy=(train_value, run_positions[i]),
                            xytext=(35, 0), textcoords='offset points',
                            ha='right', va='center', fontsize=9)

                    ax.annotate(f'{val_value:.3f}', xy=(val_value, run_positions[i]),
                            xytext=(-35, 0), textcoords='offset points',
                            ha='left', va='center', fontsize=9)

            # Add horizontal grid lines
            ax.grid(True, axis='y', linestyle='-', alpha=0.1)

            # Vertical grid lines
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

            # Set y-axis ticks and run labels
            ax.set_yticks(run_positions)
            ax.set_yticklabels([run.shortname for run in runs])

            # Set metric title with large font on the right side
            ax.text(0.99, 0.5, metric_short_name,
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))

            # Remove top, right and left spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Show only the bottom spine
            ax.spines['bottom'].set_position(('outward', 10))

            # Show major x-axis ticks
            ax.tick_params(axis='x', which='major', length=5, width=1.5)

        # Clean non-nan values
        all_values = [v for v in all_values if not np.isnan(v)]

        # Set consistent x-axis limits across all subplots
        if all_values:
            min_value = max(0, min(all_values) * 0.95)  # Don't go below 0
            max_value = max(all_values) * 1.05  # Add a little space

            for ax in axes:
                ax.set_xlim(min_value, max_value)

        # Create custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                markeredgecolor='black', markersize=10, label="Training"),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                markeredgecolor='black', markersize=10, alpha=0.6, label="Validation"),
        ]

        # Find the right location for the legend
        if n_metric_pairs > 1:
            # Place legend at the bottom
            fig.legend(handles=legend_elements,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.04),
                    ncol=2,
                    frameon=False)
        else:
            # Place legend inside the single plot
            axes[0].legend(handles=legend_elements,
                        loc='upper right',
                        frameon=False)

        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle("Training vs Validation Metrics Comparison", fontsize=16, y=0.98)

        # Add x-axis label at the bottom
        fig.text(0.5, 0.01, 'Metric Value', ha='center', fontsize=12)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.3)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig


    def plot_train_val_metrics_grid(self,
                            runs: List[ModelRun],
                            metrics: List[str],
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 10),
                            run_colors: Optional[Dict[str, str]] = None,
                            metric_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> plt.Figure:
        """
        Create a grid of small multiple bar charts, with each run in its own row.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of metrics to plot. Should contain pairs like ["Train Accuracy", "Val Accuracy"]
            title: Optional title for the plot
            figsize: Figure size (width, height)
            run_colors: Optional dictionary mapping run IDs or types to specific colors
            metric_mapping: Optional dictionary mapping run type to metric name patterns

        Returns:
            Matplotlib figure object
        """
        from matplotlib.patches import Patch

        n_runs = len(runs)
        n_metric_pairs = len(metrics) // 2

        # Adjust figsize based on number of runs and metrics
        if figsize is None:
            figsize = (3 * n_metric_pairs, 2 * n_runs)

        # Create figure with grid of subplots - one row per run
        fig, axes = plt.subplots(n_runs, n_metric_pairs, figsize=figsize,
                                sharey='row', sharex='col')

        # Handle case with single run or single metric
        if n_runs == 1 and n_metric_pairs == 1:
            axes = np.array([[axes]])
        elif n_runs == 1:
            axes = axes.reshape(1, -1)
        elif n_metric_pairs == 1:
            axes = axes.reshape(-1, 1)

        # Generate colors for runs if not provided
        if run_colors is None:
            cmap = cm.get_cmap('tab10')
            norm = Normalize(vmin=0, vmax=max(n_runs-1, 1))
            run_colors = {run.id: cmap(norm(i)) for i, run in enumerate(runs)}

        # Define colors for train vs validation
        train_alpha = 0.9
        val_alpha = 0.5
        bar_width = 0.35

        # For storing metric names for column headers
        metric_names = []

        # Process each metric pair
        for j in range(n_metric_pairs):
            train_idx = j * 2
            val_idx = j * 2 + 1

            if train_idx >= len(metrics) or val_idx >= len(metrics):
                continue

            train_metric_name = metrics[train_idx]
            val_metric_name = metrics[val_idx]

            # Get metric short name for axis label
            metric_short_name = train_metric_name.replace("Train ", "")
            if " " not in metric_short_name:
                metric_short_name = train_metric_name

            metric_names.append(metric_short_name)

            # Process each run
            for i, run in enumerate(runs):
                ax = axes[i, j]
                run_data = self.get_run_data(run)

                # Determine run type for metric mapping
                run_type = "AST" if "AST" in run.shortname else "Pretrained"

                # Get run color
                if run.id in run_colors:
                    run_color = run_colors[run.id]
                elif run_type in run_colors:
                    run_color = run_colors[run_type]
                else:
                    run_color = f"C{i}"

                # Get the appropriate metric names for this run type
                if metric_mapping and run_type in metric_mapping:
                    if train_metric_name in metric_mapping[run_type]:
                        actual_train_metric = metric_mapping[run_type][train_metric_name]
                    else:
                        actual_train_metric = train_metric_name

                    if val_metric_name in metric_mapping[run_type]:
                        actual_val_metric = metric_mapping[run_type][val_metric_name]
                    else:
                        actual_val_metric = val_metric_name
                else:
                    actual_train_metric = train_metric_name
                    actual_val_metric = val_metric_name

                # Get train value
                try:
                    train_series = get_run_metric_data(run_data, actual_train_metric)
                    train_value = float(train_series.iloc[-1]) if len(train_series) > 0 else 0
                except KeyError as e:
                    print(f"Error getting {actual_train_metric} for {run.display_name}: {e}")
                    train_value = 0

                # Get val value
                try:
                    val_series = get_run_metric_data(run_data, actual_val_metric)
                    val_value = float(val_series.iloc[-1]) if len(val_series) > 0 else 0
                except KeyError as e:
                    print(f"Error getting {actual_val_metric} for {run.display_name}: {e}")
                    val_value = 0

                # Plot bars
                ax.bar(1, train_value, width=bar_width, color=run_color, alpha=train_alpha,
                    edgecolor='black', linewidth=0.5, label="Train")
                ax.bar(2, val_value, width=bar_width, color=run_color, alpha=val_alpha,
                    edgecolor='black', linewidth=0.5, label="Val")

                # Add value labels
                ax.text(1, train_value + 0.01, f'{train_value:.3f}',
                    ha='center', va='bottom', fontsize=8)
                ax.text(2, val_value + 0.01, f'{val_value:.3f}',
                    ha='center', va='bottom', fontsize=8)

                # Set axis limits and ticks
                ax.set_xlim(0.5, 2.5)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Train", "Val"])

                # Add row labels (run names) on the first column
                if j == 0:
                    ax.set_ylabel(run.shortname, fontsize=10, rotation=0, ha='right')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Add grid
                ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add column headers (metric names)
        for j, metric_name in enumerate(metric_names):
            axes[0, j].set_title(metric_name)

        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle("Training vs Validation Metrics by Run", fontsize=16, y=0.98)

        # Add legend at the bottom
        legend_elements = [
            Patch(facecolor='gray', alpha=train_alpha, edgecolor='black', label="Train"),
            Patch(facecolor='gray', alpha=val_alpha, edgecolor='black', label="Val")
        ]
        fig.legend(handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
                frameon=False)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig

    def plot_train_val_metrics_heatmap(self,
                                runs: List[ModelRun],
                                metrics: List[str],
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8),
                                cmap: str = "magma",
                                metric_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> plt.Figure:
        """
        Create a heatmap grid showing training and validation metrics for all runs.

        Args:
            runs: List of ModelRun objects to include
            metrics: List of metrics to plot. Should contain pairs like ["Train Accuracy", "Val Accuracy"]
            title: Optional title for the plot
            figsize: Figure size (width, height)
            cmap: Colormap to use for the heatmap
            metric_mapping: Optional dictionary mapping run type to metric name patterns

        Returns:
            Matplotlib figure object
        """
        import seaborn as sns

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Number of runs and metric pairs
        n_runs = len(runs)
        n_metric_pairs = len(metrics) // 2

        # Create data structure for heatmap
        heatmap_data = np.zeros((n_runs * 2, n_metric_pairs))

        # Create row and column labels
        row_labels = []
        col_labels = []

        # Process each run and metric
        for i, run in enumerate(runs):
            run_data = self.get_run_data(run)

            # Determine run type for metric mapping
            run_type = "AST" if "AST" in run.shortname else "Pretrained"

            # Add row labels for train and val
            row_labels.append(f"{run.shortname} (Train)")
            row_labels.append(f"{run.shortname} (Val)")

            # Process each metric pair
            for j in range(n_metric_pairs):
                train_idx = j * 2
                val_idx = j * 2 + 1

                if train_idx >= len(metrics) or val_idx >= len(metrics):
                    continue

                train_metric_name = metrics[train_idx]
                val_metric_name = metrics[val_idx]

                # Get metric short name for column label
                metric_short_name = train_metric_name.replace("Train ", "")
                if " " not in metric_short_name:
                    metric_short_name = train_metric_name

                # Add column label (only once)
                if i == 0:
                    col_labels.append(metric_short_name)

                # Get the appropriate metric names for this run type
                if metric_mapping and run_type in metric_mapping:
                    if train_metric_name in metric_mapping[run_type]:
                        actual_train_metric = metric_mapping[run_type][train_metric_name]
                    else:
                        actual_train_metric = train_metric_name

                    if val_metric_name in metric_mapping[run_type]:
                        actual_val_metric = metric_mapping[run_type][val_metric_name]
                    else:
                        actual_val_metric = val_metric_name
                else:
                    actual_train_metric = train_metric_name
                    actual_val_metric = val_metric_name

                # Get train value
                try:
                    train_series = get_run_metric_data(run_data, actual_train_metric)
                    train_value = float(train_series.iloc[-1]) if len(train_series) > 0 else np.nan
                    heatmap_data[i*2, j] = train_value
                except KeyError as e:
                    print(f"Error getting {actual_train_metric} for {run.display_name}: {e}")
                    heatmap_data[i*2, j] = np.nan

                # Get val value
                try:
                    val_series = get_run_metric_data(run_data, actual_val_metric)
                    val_value = float(val_series.iloc[-1]) if len(val_series) > 0 else np.nan
                    heatmap_data[i*2 + 1, j] = val_value
                except KeyError as e:
                    print(f"Error getting {actual_val_metric} for {run.display_name}: {e}")
                    heatmap_data[i*2 + 1, j] = np.nan

        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap=cmap, fmt=".3f",
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, xticklabels=col_labels, yticklabels=row_labels)

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Training vs Validation Metrics Comparison")

        fig.tight_layout()

        return fig


    def plot_metrics_vs_dataset_size(self,
                             runs_by_size: Dict[str, List[ModelRun]],
                             metric: str,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             run_type_colors: Optional[Dict[str, str]] = None,
                             metric_mapping: Optional[Dict[str, Dict[str, str]]] = None,
                             size_order: Optional[List[str]] = None,
                             include_markers: bool = True,
                             log_scale: bool = False) -> plt.Figure:
        """
        Create a line plot showing how a metric changes with dataset size.

        Args:
            runs_by_size: Dictionary mapping dataset sizes to lists of runs
                        e.g., {"2K": [AST_2K, PRETRAINED_2K], "20K": [AST_20K, PRETRAINED_20K], ...}
            metric: Base metric name to plot (e.g., "Val Accuracy")
            title: Optional title for the plot
            figsize: Figure size (width, height)
            run_type_colors: Dictionary mapping run types to colors
                            e.g., {"AST": "red", "Pretrained": "blue"}
            metric_mapping: Optional dictionary mapping run type to metric name patterns
            size_order: Optional list to specify the order of sizes on the x-axis
                    e.g., ["2K", "20K", "100K"]
            include_markers: Whether to add markers to the data points
            log_scale: Whether to use log scale for the x-axis (useful for wide range of sizes)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Determine size order if not provided
        if size_order is None:
            # Try to extract numbers from size strings and sort
            def extract_num(s):
                return int(''.join(filter(str.isdigit, s)))

            try:
                size_order = sorted(runs_by_size.keys(), key=extract_num)
            except:
                # Fallback to default sort
                size_order = sorted(runs_by_size.keys())

        # Collect runs by type to plot lines connecting same type
        run_types = set()
        runs_by_type = {}

        for size in size_order:
            for run in runs_by_size.get(size, []):
                # Determine run type
                if "AST" in run.shortname:
                    run_type = "AST"
                elif "PRETRAINED" in run.shortname:
                    run_type = "Pretrained"
                else:
                    run_type = "Other"

                run_types.add(run_type)

                if run_type not in runs_by_type:
                    runs_by_type[run_type] = {}

                runs_by_type[run_type][size] = run

        # Plot lines for each run type
        for run_type in sorted(run_types):
            sizes = []
            values = []

            for size in size_order:
                if size in runs_by_type[run_type]:
                    run = runs_by_type[run_type][size]
                    run_data = self.get_run_data(run)

                    # Get the appropriate metric name for this run type
                    if run_type in metric_mapping and metric in metric_mapping[run_type]:
                        actual_metric_name = metric_mapping[run_type][metric]
                    else:
                        actual_metric_name = metric

                    try:
                        # Try to get metric info
                        try:
                            metric_info = get_metric(actual_metric_name)
                            display_name = metric_info.display_name
                        except:
                            display_name = actual_metric_name

                        metric_series = get_run_metric_data(run_data, actual_metric_name)

                        if len(metric_series) == 0:
                            print(f"Warning: No data found for metric '{actual_metric_name}' in run '{run.display_name}'")
                            continue

                        # Get the final value
                        final_value = float(metric_series.iloc[-1])

                        sizes.append(size)
                        values.append(final_value)

                    except KeyError as e:
                        print(f"Error getting {actual_metric_name} for {run.display_name}: {e}")

            # Plot the line if we have data points
            if sizes and values:
                # Get color for this run type
                color = run_type_colors.get(run_type, "gray")

                # Create market style
                marker = 'o' if include_markers else None

                # Plot the line
                ax.plot(sizes, values, marker=marker, label=run_type,
                    color=color, linewidth=2, markersize=8)

                # Add data labels
                for x, y in zip(sizes, values):
                    ax.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

        # Set plot properties
        if title:
            ax.set_title(title)
        else:
            try:
                metric_display = get_metric(metric).display_name
            except:
                metric_display = metric
            ax.set_title(f"{metric_display} vs Dataset Size")

        # Set axis labels
        ax.set_xlabel("Dataset Size")

        try:
            y_label = get_metric(metric).display_name
        except:
            y_label = metric
        ax.set_ylabel(y_label)

        # Set x-axis to log scale if requested
        if log_scale:
            ax.set_xscale('log')

        # Set x-ticks to the dataset sizes
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend()

        fig.tight_layout()

        return fig


    def save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save a figure to a file in the output directory"""
        # Clean up filename to be filesystem-friendly
        clean_filename = filename.replace(" ", "_").replace(":", "").replace("?", "").replace("*", "").replace("|", "")
        
        # Ensure it has a .png extension
        if not clean_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            clean_filename += '.png'
            
        output_path = self.output_dir / clean_filename
        fig.savefig(output_path)
        plt.close(fig)
        
        return output_path
