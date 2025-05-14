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
confusionComparison = True
finalComparisons = True
sizeVsPerf = True

run_colors = {
    AST_2K.id: "#CC2222",           # Dark Red
    AST_20K.id: "#CC9900",          # Dark Yellow
    AST_100K.id: "#0099AA",         # Dark Blue
    PRETRAINED_2K.id: "#008F4F",    # Dark Green
    PRETRAINED_20K.id: "#CC0088",   # Dark Pink
    PRETRAINED_100K.id: "#f26849"   # Dark Orange
}

run_type_colors = {
    "AST": "#CC2222",               # Dark Red
    "Pretrained": "#008F4F"         # Dark Green
}

# so I got this idea a bit too late, and doesnt make sense to go back and implement it for the older methods
# we're never gonna use this project again so whatever x)
metric_mapping = {
    "AST": {
        "Accuracy": "Train Accuracy",
        "Precision": "Train Precision",
        "Recall": "Train Recall",
        "F1 Score": "Train F1 Score",
        "Loss": "Train Loss",
        "Train Accuracy": "Train Accuracy",
        "Train Precision": "Train Precision",
        "Train Recall": "Train Recall",
        "Train F1 Score": "Train F1 Score",
        "Train Loss": "Train Loss",
        "Val Accuracy": "Val Accuracy",
        "Val Precision": "Val Precision",
        "Val Recall": "Val Recall",
        "Val F1 Score": "Val F1 Score",
        "Val Loss": "Val Loss"
    },
    "Pretrained": {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1 Score": "F1 Score",
        "Loss": "Loss",
        "Train Accuracy": "Accuracy",
        "Train Precision": "Precision",
        "Train Recall": "Recall",
        "Train F1 Score": "F1 Score",
        "Train Loss": "Loss",
        "Val Accuracy": "Val Accuracy",
        "Val Precision": "Val Precision",
        "Val Recall": "Val Recall",
        "Val F1 Score": "Val F1 Score",
        "Val Loss": "Val Loss"
    }
}

runs_by_size = {
    "2K": [AST_2K, PRETRAINED_2K],
    "20K": [AST_20K, PRETRAINED_20K],
    "100K": [AST_100K, PRETRAINED_100K]
}


if sizeVsPerf:

    metrics = [
        "Train Accuracy",
        "Train Precision",
        "Train Recall",
        "Train F1 Score",
        "Train Loss",
        "Val Accuracy",
        "Val Precision",
        "Val Recall",
        "Val F1 Score",
        "Val Loss"
    ]

    for metric in metrics:
        try:
            metric_info = get_metric(metric)
            metric_display = metric_info.display_name
        except:
            metric_display = metric

        size_performance_plot = visualizer.plot_metrics_vs_dataset_size(
            runs_by_size=runs_by_size,
            metric=metric,
            title=f"{metric_display} vs Dataset Size",
            figsize=(10, 6),
            run_type_colors=run_type_colors,
            metric_mapping=metric_mapping,
            size_order=["2K", "20K", "100K"],
            include_markers=True
        )

        # Save the figure
        metric_filename = metric.lower().replace(" ", "_").replace("/", "_")
        visualizer.save_figure(size_performance_plot, f"SIZE_VS_{metric_filename}.png")



if finalComparisons:
    print("Creating final metrics bar plots...")

    run_groups = [
        (ALL_RUNS, "Train Accuracy", "All Models"),
        (ALL_RUNS, "Val Accuracy", "All Models"),
        (AST_RUNS, "Train Accuracy", "AST Models"),
        (AST_RUNS, "Val Accuracy", "AST Models"),
        (PRETRAINED_RUNS, "Train Accuracy", "Pretrained Models"),
        (PRETRAINED_RUNS, "Val Accuracy", "Pretrained Models"),
        (ALL_RUNS, "Train Precision", "All Models"),
        (ALL_RUNS, "Val Precision", "All Models"),
        (AST_RUNS, "Train Precision", "AST Models"),
        (AST_RUNS, "Val Precision", "AST Models"),
        (PRETRAINED_RUNS, "Train Precision", "Pretrained Models"),
        (PRETRAINED_RUNS, "Val Precision", "Pretrained Models"),
        (ALL_RUNS, "Train Recall", "All Models"),
        (ALL_RUNS, "Val Recall", "All Models"),
        (AST_RUNS, "Train Recall", "AST Models"),
        (AST_RUNS, "Val Recall", "AST Models"),
        (PRETRAINED_RUNS, "Train Recall", "Pretrained Models"),
        (PRETRAINED_RUNS, "Val Recall", "Pretrained Models"),
        (ALL_RUNS, "Train F1 Score", "All Models"),
        (ALL_RUNS, "Val F1 Score", "All Models"),
        (AST_RUNS, "Train F1 Score", "AST Models"),
        (AST_RUNS, "Val F1 Score", "AST Models"),
        (PRETRAINED_RUNS, "Train F1 Score", "Pretrained Models"),
        (PRETRAINED_RUNS, "Val F1 Score", "Pretrained Models")
    ]


    for run_group, metric_name, display_name in run_groups:
        final_acc_comparison = visualizer.plot_final_metrics(
            runs=run_group,
            metrics=[metric_name],
            title=f"Final {metric_name} Across {display_name}",
            figsize=(12, 8),
            run_colors=run_colors,
            metric_mapping=metric_mapping
        )
        visualizer.save_figure(final_acc_comparison, f"BarChart_{display_name}_{metric_name}.png")




if confusionComparison:
    comparisons = [
        (AST_2K, PRETRAINED_2K, "train", "2K"),
        (AST_2K, PRETRAINED_2K, "val", "2K"),
        (AST_20K, PRETRAINED_20K, "train", "20K"),
        (AST_20K, PRETRAINED_20K, "val", "20K"),
        (AST_100K, PRETRAINED_100K, "train", "100K"),
        (AST_100K, PRETRAINED_100K, "val", "100K"),
    ]

    for run1, run2, metric_type, size in comparisons:
        pretty_title = f"{metric_type.capitalize()} Confusion Matrix - {size} (AST vs Pretrained) "
        fig_conf_train = visualizer.plot_confusion_matrix_between_runs(
                    run1=run1,
                    run2=run2,
                    metric_type=metric_type,
                    title=pretty_title,
                    figsize=(8, 6)
                )
        visualizer.save_figure(fig_conf_train, f"ConfusionMatrix_{size}_{metric_type}.png")


if individualFigures:
    print("#########################")
    print("Creating figures PER RUN")
    print("#########################")

    run_data = [
        (AST_2K, "AST"),
        (AST_20K, "AST"),
        (AST_100K, "AST"),
        (PRETRAINED_2K, "pretrained"),
        (PRETRAINED_20K, "pretrained"),
        (PRETRAINED_100K, "pretrained"),
    ]

    for run, run_type in run_data:
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


        visualizer.save_figure(fig_conf_train, f"ConfusionMatrix__{run.shortname}_train.png")
        visualizer.save_figure(fig_conf_val, f"ConfusionMatrix__{run.shortname}_val.png")
        visualizer.save_figure(fig_conf_both, f"ConfusionMatrix__{run.shortname}_comparison.png")


        print("Making train/val loss graphs")
        fig_loss = visualizer.compare_metrics(
            run=run,
            metrics=["Train Loss", "Val Loss", "Loss"],
            title=f"Training vs Validation Loss - {run.display_name}",
            figsize=(10, 6),
            smoothing=5
        )
        visualizer.save_figure(fig_loss, f"LineGraph_{run.shortname}_loss_train_val.png")

        print("Making classification metrics graphs")

        for metric in [
            ("Train Accuracy", "Val Accuracy", "Accuracy"),
            ("Train Precision", "Val Precision", "Precision"),
            ("Train Recall", "Val Recall", "Recall"),
            ("Train F1 Score", "Val F1 Score", "F1 Score")
        ]:
            # no I dont think this is good code, but I am too lazy to fix it
            train_metric = metric[2] if run_type.lower() == "pretrained" else metric[0]
            fig_class_metrics = visualizer.compare_metrics(
                run=run,
                metrics=[train_metric, metric[1]],
                title=f"{train_metric} vs {metric[1]} - {run.display_name}",
                figsize=(10, 6),
                smoothing=5
            )
            pretty_name = f"{metric[2].lower().replace(' ','_')}_train_val"
            visualizer.save_figure(fig_class_metrics, f"LineGraph_{run.shortname}_{pretty_name}.png")


if comparisonFigures:
    print("############################")
    print("Creating figures ACROSS RUNS")
    print("############################")

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
    visualizer.save_figure(ast_loss_comparison, "LineGraph_AST_loss_train_val.png")

    # Compare accuracy for all AST models
    ast_acc_comparison = visualizer.compare_metrics_across_runs(
        runs=AST_RUNS,
        metrics=["Train Accuracy", "Val Accuracy"],
        title="Training & Validation Accuracy Comparison - AST Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors
    )
    visualizer.save_figure(ast_acc_comparison, "LineGraph_AST_accuracy_train_val.png")

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
    visualizer.save_figure(pretrained_loss_comparison, "LineGraph_Pretrained_loss_train_val.png")

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
    visualizer.save_figure(pretrained_acc_comparison, "LineGraph_Pretrained_accuracy_train_val.png")

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
            comparison_figure = visualizer.compare_metrics_across_runs(
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
            visualizer.save_figure(comparison_figure, f"LineGraph_{size_label}_{name}_train_val.png")


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
    visualizer.save_figure(all_models_comparison, "LineGraph_AllModels_accuracy_train_val.png")

    f1_comparison = visualizer.compare_metrics_across_runs(
        runs=[AST_2K, AST_20K, AST_100K, PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K],
        metrics=["Train F1 Score", "Val F1 Score", "Val F1 Score"],
        title="Validation F1 Score Comparison - All Models",
        figsize=(14, 8),
        smoothing=5,
        run_colors=run_colors,
        line_styles = {
            "F1 Score": "solid",
            "Train F1 Score": "solid",
            "Val F1 Score": "dashed"
        }
    )
    visualizer.save_figure(f1_comparison, "LineGraph_AllModels_F1Score_train_val.png")

print("All comparison plots created successfully!")
























