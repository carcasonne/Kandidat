import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from metric_categories import get_metric
from pathlib import Path
from metric_categories import get_metric, MetricCategory
from visualizer import Visualizer, get_run_metric_data

from model_runs import (
    ModelRun,
    AST_RUNS,
    PRETRAINED_RUNS,
    ALL_RUNS,
    AST_2K,
    AST_20K,
    AST_100K,
    PRETRAINED_2K,
    PRETRAINED_20K,
    PRETRAINED_100K,
    get_runs_by_size,
    DataSize,
)

from wandb_login import login
login()

visualizer = Visualizer(output_dir=Path("custom_plots"))

individualFigures = False
comparisonFigures = True

if individualFigures:
    print("#########################")
    print("Creating figures PER RUN")
    print("#########################")
    for run in [AST_100K]:
        print(f"Creating figures for {run.display_name}...")

        print("Making confusion matrices")
        fig_conf_train = visualizer.plot_confusion_matrix(
                run=run,
                metric_name = "train_conf_mat",
                title=f"Training Confusion Matrix - {run.display_name}",
                figsize=(8, 6),
                both=False
            )
        fig_conf_val = visualizer.plot_confusion_matrix(
                run=run,
                metric_name = "val_conf_mat",
                title=f"Validation Confusion Matrix - {run.display_name}",
                figsize=(8, 6),
                both=False
            )
        fig_conf_both = visualizer.plot_confusion_matrix(
                run=run,
                title=f"Training/Validation Confusion Matrix - {run.display_name}",
                figsize=(8, 6),
                both=True,
            )


        visualizer.save_figure(fig_conf_train, f"{run.shortname}_train_confusion_matrix.png")
        visualizer.save_figure(fig_conf_val, f"{run.shortname}_val_confusion_matrix.png")
        visualizer.save_figure(fig_conf_both, f"{run.shortname}_confusion_matrixes.png")


        print("Making train/val loss graphs")
        fig_loss = visualizer.compare_metrics(
            run=run,
            metrics=["Train Loss", "Val Loss"],
            title=f"Training vs Validation Loss - {run.display_name}",
            figsize=(10, 6),
            smoothing=5
        )
        visualizer.save_figure(fig_loss, f"{run.shortname}_train_val_loss.png")

        print("Making classification metrics graphs")

        for metric in [
            ("Train Accuracy","Val Accuracy"),
            ("Train Precision", "Val Precision"),
            ("Train Recall", "Val Recall"),
            ("Train F1 Score", "Val F1 Score")
        ]:
            fig_class_metrics = visualizer.compare_metrics(
                run=run,
                metrics=[metric[0], metric[1]],
                title=f"{metric[0]} vs {metric[1]} - {run.display_name}",
                figsize=(10, 6),
                smoothing=5
            )
            pretty_name = f"{metric[0].lower().replace(' ','_')}"
            visualizer.save_figure(fig_class_metrics, f"{run.shortname}_{pretty_name}.png")


if comparisonFigures:
    print("############################")
    print("Creating figures ACROSS RUNS")
    print("############################")

    run_colors = {
        AST_2K.id: "#008F4F",           # Dark Green
        AST_20K.id: "#0099AA",          # Dark Blue
        AST_100K.id: "#CC9900",         # Dark Yellow
        PRETRAINED_2K.id: "#CC2222",    # Dark Red
        PRETRAINED_20K.id: "#CC0088",   # Dark Pink
        PRETRAINED_100K.id: "#6600CC"   # Dark Purple
    }

    for run_group in [[AST_2K, AST_20K, AST_100K]]:
        fig_train_loss_comparison = visualizer.compare_runs_with_distinct_colors(
            runs=run_group,
            metric="Train Loss",
            title="Training Loss Comparison Across Runs",
            figsize=(12, 6),
            smoothing=5,
            run_colors=run_colors  # Pass your color mapping
        )
        run_names = "_".join(run.shortname for run in run_group)
        visualizer.save_figure(fig_train_loss_comparison, f"{run_names}_comparison_train_loss.png")

    # Ok this is kinda stupid, but we have given distinct names for metrics in AST and Pretrained:
    # AST: Train Loss, Val Loss
    # Pretrained: Loss, Val Loss
    # so easiest to just play aorund this rather than change entire system
    for run_group in [[PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K]]:
        fig_train_loss_comparison = visualizer.compare_runs_with_distinct_colors(
            runs=run_group,
            metrics="Loss",
            title="Training Loss Comparison Across Runs",
            figsize=(12, 6),
            smoothing=5,
            run_colors=run_colors
        )
        run_names = "_".join(run.shortname for run in run_group)
        visualizer.save_figure(fig_train_loss_comparison, f"{run_names}_comparison_train_loss.png")
























