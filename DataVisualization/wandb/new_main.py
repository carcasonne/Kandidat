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

individualFigures = True
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
        AST_2K.id: "#CC2222",           # Dark Red
        AST_20K.id: "#0099AA",          # Dark Blue
        AST_100K.id: "#CC9900",         # Dark Yellow
        PRETRAINED_2K.id: "#008F4F",    # Dark Green
        PRETRAINED_20K.id: "#CC0088",   # Dark Pink
        PRETRAINED_100K.id: "#f26849"   # Dark Orange
    }

    AST_RUNS = [AST_2K, AST_20K, AST_100K]
    PRETRAINED_RUNS = [PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K]

    # Compare train/vall loss for all AST models
    ast_loss_comparison = visualizer.compare_metrics_across_runs(
        runs=AST_RUNS,  #
        metrics=["Train Loss", "Val Loss"],
        title="Training & Validation Loss Comparison - AST Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors
    )
    visualizer.save_figure(ast_loss_comparison, "AST_ALL_MODELS_loss_comparison.png")

    # Compare accuracy for all AST models
    ast_acc_comparison = visualizer.compare_metrics_across_runs(
        runs=AST_RUNS,
        metrics=["Train Accuracy", "Val Accuracy"],
        title="Training & Validation Accuracy Comparison - AST Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors
    )
    visualizer.save_figure(ast_acc_comparison, "AST_ALL_MODELS_accuracy_comparison.png")

    # Compare train/vall loss for all Pretrained models
    pretrained_loss_comparison = visualizer.compare_metrics_across_runs(
        runs=PRETRAINED_RUNS,
        metrics=["Loss", "Val Loss"],
        title="Training & Validation Loss Comparison - Pretrained Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors,
        line_styles = {
            "Loss": "solid",
            "Val Loss": "dashed"
        }
    )
    visualizer.save_figure(pretrained_loss_comparison, "PRETRAINED_ALL_MODELS_loss_comparison.png")

    # Compare accuracy for all Pretrained models
    pretrained_acc_comparison = visualizer.compare_metrics_across_runs(
        runs=PRETRAINED_RUNS,
        metrics=["Accuracy", "Val Accuracy"],
        title="Training & Validation Accuracy Comparison - Pretrained Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors,
        line_styles = {
            "Accuracy": "solid",
            "Val Accuracy": "dashed"
        }
    )
    visualizer.save_figure(pretrained_acc_comparison, "PRETRAINED_ALL_MODELS_accuracy_comparison.png")

    # Compare a metric across models for same size
    size_pairs = [
        (AST_2K, PRETRAINED_2K, "2K"),
        (AST_20K, PRETRAINED_20K, "20K"),
        (AST_100K, PRETRAINED_100K, "100K")
    ]

    # NOTE!!!
    # This is a bit scuffed, since AST will have "Train <metric>" and "Val <metric>",
    # but Pretrained will have "<metric>" and "Val <metric>".
    # So really 2 * <no. of metrics in group> lookups are performed per iteration
    # and half of those calls will fail, as that metric does not exist for this model

    metric_groups = [
        ("Train Precision", "Val Precision", "Precision"),
        ("Train Accuracy", "Val Accuracy", "Accuracy"),
        ("Train Recall", "Val Recall", "Recall"),
        ("Train F1", "Val F1", "F1"),
        ("Train Loss", "Val Loss", "Loss"),
    ]

    # For each pair, create loss and accuracy comparisons
    for ast_model, pretrained_model, size_label in size_pairs:
        for train, val, name in metric_groups:
            precision_comparison = visualizer.compare_metrics_across_runs(
                runs=[ast_model, pretrained_model],
                metrics=[train, val, name],
                title=f"{name} Comparison - {size_label} Models (AST vs Pretrained)",
                figsize=(14, 8),
                smoothing=5,
                run_colors=run_colors,
                line_styles = {
                    name: "solid",
                    train: "solid",
                    val: "dashed"
                }
            )
            visualizer.save_figure(precision_comparison, f"COMPARISON_{size_label}_{name}_precision.png")


    # Add direct AST vs Pretrained comparison across all sizes
    print("Creating AST vs Pretrained comparison across all sizes...")
    all_models_comparison = visualizer.compare_metrics_across_runs(
        runs=[AST_2K, AST_20K, AST_100K, PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K],
        metrics=["Train Accuracy", "Val Accuracy", "Accuracy"],
        title="Validation Accuracy Comparison - All Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors,
        line_styles = {
            "Accuracy": "solid",
            "Train Accuracy": "solid",
            "Val Accuracy": "dashed"
        }
    )
    visualizer.save_figure(all_models_comparison, "ALL_MODELS_accuracy_comparison.png")

    # Similarly for F1 Score if you have it
    f1_comparison = visualizer.compare_metrics_across_runs(
        runs=[AST_2K, AST_20K, AST_100K, PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K],
        metrics=["Train F1", "Val F1", "Val F1"],
        title="Validation F1 Score Comparison - All Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors,
        line_styles = {
            "F1": "solid",
            "Train F1": "solid",
            "Val F1": "dashed"
        }
    )
    visualizer.save_figure(f1_comparison, "ALL_MODELS_f1_comparison.png")

print("All comparison plots created successfully!")
























