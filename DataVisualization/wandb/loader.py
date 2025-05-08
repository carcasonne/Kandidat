import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


from typing import List

def set_theme():
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

def generate_graphs_from_run(entity: str, project: str, run_id: str, metrics: List[str], displayname: str, output_dir: Path = Path("plots")):
    set_theme()
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Retrieve full unsampled history
    history = run.scan_history()
    history_df = pd.DataFrame(list(history))

    # Create output directory for this run
    run_output_dir = output_dir / displayname
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare DataFrame with only the metrics we care about
    plot_data = {}
    for metric in metrics:
        if metric in history_df.columns:
            series = history_df[metric].dropna().reset_index(drop=True)
            plot_data[metric] = series

    if not plot_data:
        print(f"No metrics found for run {run_id}")
        return

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    for name, values in plot_data.items():
        sns.lineplot(data=values, label=name)

    plt.title(f"Metrics Over Time — {displayname}")
    plt.xlabel("Step (logged index)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()

    # Save to file
    output_path = run_output_dir / "metrics.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to: {output_path}")
