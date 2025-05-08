import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Optional: import your metric constants if you're using them
try:
    import metric_names as mn
    METRICS = [mn.TRAIN_ACCURACY, mn.VAL_ACCURACY]
except ImportError:
    # fallback if not using metric_names.py
    METRICS = ["Train Accuracy", "Val Accuracy", "Train Loss"]

def get_visuals_from_run(entity: str, project: str, run_id: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Fetch full unsampled history
    history = run.scan_history()
    history_list = list(history)
    history_df = pd.DataFrame(history_list)

    print("=== Available columns ===")
    print(history_df.columns.tolist())

    print("\n=== First 10 rows of history ===")
    print(history_df.head(10))

    print("\n=== Selected metrics (first 10 rows) ===")
    for metric in METRICS:
        if metric in history_df.columns:
            print(f"\n-- {metric} --")
            print(history_df[[metric]].dropna().head(10))
        else:
            print(f"\n-- {metric} not found in columns --")

    # Plot available metrics
    for metric in METRICS:
        if metric in history_df.columns:
            metric_data = history_df[[metric]].dropna()
            if not metric_data.empty:
                plt.plot(metric_data.index, metric_data[metric], label=metric)

    plt.legend()
    plt.title("Metrics Over Time")
    plt.xlabel("Step Index")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
