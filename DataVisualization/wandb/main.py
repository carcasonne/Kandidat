import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from wandb_login import login
import wandb_runs
import loader

login()

entity = "Holdet_thesis"
project = "Kandidat-AST"

images_AST2K = loader.get_visuals_from_run(entity, project, wandb_runs.AST_2K, Path("wandb_metric_plots"))




