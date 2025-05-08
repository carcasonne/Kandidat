import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import metric_names as mn

from wandb_login import login
import wandb_runs
import loader

login()

entity = "Holdet_thesis"
project_1 = "Kandidat-AST"
project_2 = "Kandidat-Pre-trained"

runs_1 = [
    wandb_runs.AST_2K,
    wandb_runs.AST_20K,
    wandb_runs.AST_100K,
]

runs_2 = [
    wandb_runs.PRETRAINED_2K,
    wandb_runs.PRETRAINED_20K,
    wandb_runs.PRETRAINED_100K,
]

for run in runs_1:
    metrics = [mn.TRAIN_ACCURACY, mn.VAL_ACCURACY]
    loader.generate_graphs_from_run(
        entity, project_1, run.id, metrics, run.display_name, Path("wandb_metric_plots")
    )

for run in runs_2:
    metrics = ["Accuracy", mn.VAL_ACCURACY]
    loader.generate_graphs_from_run(
        entity, project_2, run.id, metrics, run.display_name, Path("wandb_metric_plots")
    )
