import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = {
    "train_loss": ("Train loss", "lower"),
    "val_loss": ("Validation loss", "lower"),
    "precision": ("Precision", "higher"),
    "recall": ("Recall", "higher"),
    "f1": ("F1", "higher"),
    "map50": ("mAP@50", "higher"),
    "map50_95": ("mAP@50:95", "higher"),
}

MODEL_ORDER = [
    "YOLOv8-S",
    "YOLOv8-M",
    "YOLO11-S",
    "YOLO11-M",
    "RF-DETR-S",
    "RF-DETR-M",
]

MODEL_COLORS = {
    "YOLOv8-S": "#1f77b4",
    "YOLOv8-M": "#0b3d91",
    "YOLO11-S": "#2ca02c",
    "YOLO11-M": "#0b6623",
    "RF-DETR-S": "#ff7f0e",
    "RF-DETR-M": "#d62728",
}

CLASS_AP_PREFIX = "class_ap_"


def read_summary(summary_path: Path) -> list[dict]:
    rows = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No successful runs found in {summary_path}")
    return rows


def model_display_name(row: dict) -> str:
    family = row["family"]
    model = str(row["model"])
    if family == "yolo":
        stem = Path(model).stem.lower()
        if stem == "yolov8s":
            return "YOLOv8-S"
        if stem == "yolov8m":
            return "YOLOv8-M"
        if stem == "yolo11s":
            return "YOLO11-S"
        if stem == "yolo11m":
            return "YOLO11-M"
        return stem.upper()
    if family == "rfdetr":
        suffix = {"small": "S", "medium": "M", "large": "L"}.get(model, model.upper())
        return f"RF-DETR-{suffix}"
    return row["run_name"]


def last_non_null(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[-1]


def load_yolo_history(row: dict) -> pd.DataFrame:
    csv_path = Path(row["output_dir"]) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing YOLO results.csv for {row['run_name']}: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    out = pd.DataFrame()
    out["raw_epoch"] = df["epoch"]
    out["epoch"] = range(1, len(df) + 1)
    out["train_loss"] = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
    out["val_loss"] = df[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)
    out["precision"] = df["metrics/precision(B)"]
    out["recall"] = df["metrics/recall(B)"]
    out["map50"] = df["metrics/mAP50(B)"]
    out["map50_95"] = df["metrics/mAP50-95(B)"]
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"] + 1e-12)
    out["time_seconds"] = df["time"] if "time" in df.columns else pd.NA
    return add_run_columns(out, row)


def load_rfdetr_history(row: dict) -> pd.DataFrame:
    csv_path = Path(row["output_dir"]) / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RF-DETR metrics.csv for {row['run_name']}: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    grouped = (
        df.sort_values(["epoch", "step"])
        .groupby("epoch", as_index=False)[numeric_cols]
        .agg(last_non_null)
    )

    out = pd.DataFrame()
    out["raw_epoch"] = grouped["epoch"]
    out["epoch"] = range(1, len(grouped) + 1)
    out["train_loss"] = grouped.get("train/loss", pd.NA)
    out["val_loss"] = grouped.get("val/loss", pd.NA)
    out["precision"] = grouped.get("val/precision", pd.NA)
    out["recall"] = grouped.get("val/recall", pd.NA)
    out["map50"] = grouped.get("val/mAP_50", pd.NA)
    out["map50_95"] = grouped.get("val/mAP_50_95", pd.NA)
    out["f1"] = grouped.get("val/F1", pd.NA)
    missing_f1 = out["f1"].isna() & out["precision"].notna() & out["recall"].notna()
    out.loc[missing_f1, "f1"] = (
        2
        * out.loc[missing_f1, "precision"]
        * out.loc[missing_f1, "recall"]
        / (out.loc[missing_f1, "precision"] + out.loc[missing_f1, "recall"] + 1e-12)
    )

    for col in grouped.columns:
        if col.startswith("val/AP/"):
            class_name = col.split("/", 2)[2].lower()
            out[f"{CLASS_AP_PREFIX}{class_name}"] = grouped[col]

    return add_run_columns(out, row)


def add_run_columns(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df.insert(0, "model_name", model_display_name(row))
    df.insert(1, "family", row["family"])
    df.insert(2, "run_name", row["run_name"])
    df["dataset_resolution"] = row.get("resolution")
    df["input_resolution"] = row.get("input_resolution")
    df["training_seconds"] = row.get("seconds")
    df["best_weights"] = row.get("best_weights")
    return df


def load_all_histories(summary_rows: list[dict]) -> pd.DataFrame:
    histories = []
    for row in summary_rows:
        if row["family"] == "yolo":
            histories.append(load_yolo_history(row))
        elif row["family"] == "rfdetr":
            histories.append(load_rfdetr_history(row))
        else:
            raise ValueError(f"Unknown family: {row['family']}")
    history = pd.concat(histories, ignore_index=True)
    metric_cols = list(METRIC_SPECS)
    for col in metric_cols:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history["model_name"] = pd.Categorical(history["model_name"], categories=MODEL_ORDER, ordered=True)
    return history.sort_values(["model_name", "epoch"]).reset_index(drop=True)


def final_metrics(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in history.groupby("model_name", observed=True):
        row = {
            "model_name": str(model_name),
            "family": group["family"].iloc[0],
            "run_name": group["run_name"].iloc[0],
            "dataset_resolution": group["dataset_resolution"].iloc[0],
            "input_resolution": group["input_resolution"].iloc[0],
            "epochs": int(group["epoch"].max()),
            "training_hours": float(group["training_seconds"].iloc[0]) / 3600,
            "best_weights": group["best_weights"].iloc[0],
        }

        for metric, (_, direction) in METRIC_SPECS.items():
            values = group[["epoch", metric]].dropna()
            if values.empty:
                row[f"final_{metric}"] = pd.NA
                row[f"best_{metric}"] = pd.NA
                row[f"best_{metric}_epoch"] = pd.NA
                continue
            final_value = values.iloc[-1][metric]
            best_idx = values[metric].idxmax() if direction == "higher" else values[metric].idxmin()
            best = values.loc[best_idx]
            row[f"final_{metric}"] = final_value
            row[f"best_{metric}"] = best[metric]
            row[f"best_{metric}_epoch"] = int(best["epoch"])

        rows.append(row)

    result = pd.DataFrame(rows)
    result["model_name"] = pd.Categorical(result["model_name"], categories=MODEL_ORDER, ordered=True)
    return result.sort_values("model_name").reset_index(drop=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_curve(history: pd.DataFrame, metric: str, output_dir: Path) -> None:
    title, _ = METRIC_SPECS[metric]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model_name in MODEL_ORDER:
        group = history[history["model_name"].astype(str) == model_name]
        values = group[["epoch", metric]].dropna()
        if values.empty:
            continue
        ax.plot(
            values["epoch"],
            values[metric],
            marker="o",
            markersize=3.5,
            linewidth=2,
            label=model_name,
            color=MODEL_COLORS.get(model_name),
        )

    ax.set_title(f"{title} by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.legend(ncol=2)
    save_figure(fig, output_dir, f"curve_{metric}")


def plot_metric_grid(history: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["train_loss", "val_loss", "map50", "map50_95", "precision", "recall", "f1"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        title, _ = METRIC_SPECS[metric]
        for model_name in MODEL_ORDER:
            group = history[history["model_name"].astype(str) == model_name]
            values = group[["epoch", metric]].dropna()
            if values.empty:
                continue
            ax.plot(
                values["epoch"],
                values[metric],
                linewidth=1.8,
                label=model_name,
                color=MODEL_COLORS.get(model_name),
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.12)
    save_figure(fig, output_dir, "curves_overview_grid")


def plot_final_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    bar_metrics = ["best_map50", "best_map50_95", "best_precision", "best_recall", "best_f1"]
    labels = {
        "best_map50": "Best mAP@50",
        "best_map50_95": "Best mAP@50:95",
        "best_precision": "Best precision",
        "best_recall": "Best recall",
        "best_f1": "Best F1",
    }

    fig, axes = plt.subplots(len(bar_metrics), 1, figsize=(9, 12), sharex=True)
    x = range(len(summary))
    colors = [MODEL_COLORS.get(str(model), "#777777") for model in summary["model_name"]]
    for ax, metric in zip(axes, bar_metrics):
        values = pd.to_numeric(summary[metric], errors="coerce")
        ax.bar(x, values, color=colors)
        ax.set_ylabel(labels[metric])
        ax.set_ylim(0, max(1.0, float(values.max(skipna=True)) * 1.08))
        for idx, value in enumerate(values):
            if pd.notna(value):
                ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(summary["model_name"].astype(str), rotation=25, ha="right")
    save_figure(fig, output_dir, "best_metric_bars")


def plot_training_time(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    values = summary["training_hours"]
    x = range(len(summary))
    colors = [MODEL_COLORS.get(str(model), "#777777") for model in summary["model_name"]]
    ax.bar(x, values, color=colors)
    ax.set_title("Training time")
    ax.set_ylabel("Hours")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["model_name"].astype(str), rotation=25, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.2f}h", ha="center", va="bottom", fontsize=8)
    save_figure(fig, output_dir, "training_time")


def plot_precision_recall_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    for _, row in summary.iterrows():
        model_name = str(row["model_name"])
        precision = row["best_precision"]
        recall = row["best_recall"]
        if pd.isna(precision) or pd.isna(recall):
            continue
        ax.scatter(
            recall,
            precision,
            s=95,
            color=MODEL_COLORS.get(model_name),
            edgecolor="black",
            linewidth=0.5,
            label=model_name,
        )
        ax.annotate(model_name, (recall, precision), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Precision-recall trade-off")
    ax.set_xlabel("Best recall")
    ax.set_ylabel("Best precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, output_dir, "precision_recall_tradeoff")


def plot_rfdetr_class_ap(history: pd.DataFrame, output_dir: Path) -> None:
    class_cols = [col for col in history.columns if col.startswith(CLASS_AP_PREFIX)]
    if not class_cols:
        return

    rows = []
    for model_name, group in history[history["family"] == "rfdetr"].groupby("model_name", observed=True):
        last = group.sort_values("epoch").iloc[-1]
        for col in class_cols:
            value = last.get(col)
            if pd.notna(value):
                rows.append(
                    {
                        "model_name": str(model_name),
                        "class_name": col.removeprefix(CLASS_AP_PREFIX).replace("_", " ").title(),
                        "ap": value,
                    }
                )

    if not rows:
        return

    df = pd.DataFrame(rows)
    classes = sorted(df["class_name"].unique())
    fig, axes = plt.subplots(len(classes), 1, figsize=(8.5, max(2.2 * len(classes), 5.0)), sharex=True)
    if len(classes) == 1:
        axes = [axes]
    for ax, class_name in zip(axes, classes):
        sub = df[df["class_name"] == class_name].set_index("model_name").reindex(MODEL_ORDER).dropna()
        ax.bar(sub.index.astype(str), sub["ap"], color=[MODEL_COLORS.get(str(m), "#777777") for m in sub.index])
        ax.set_title(class_name)
        ax.set_ylabel("AP")
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("RF-DETR model")
    save_figure(fig, output_dir, "rfdetr_per_class_ap")


def write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "model_name",
        "family",
        "input_resolution",
        "epochs",
        "training_hours",
        "best_map50",
        "best_map50_95",
        "best_precision",
        "best_recall",
        "best_f1",
    ]
    table = summary[cols].copy()
    for col in table.columns:
        if col.startswith("best_") or col == "training_hours":
            table[col] = pd.to_numeric(table[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.astype(str).itertuples(index=False, name=None)
    ]
    output_path.write_text(
        "# Training Curves Report\n\n"
        "Generated from `runs/train_compare/summary.jsonl`.\n\n"
        "## Final/Best Metrics\n\n"
        + "\n".join([header, separator, *body])
        + "\n",
        encoding="utf-8",
    )


def build_report(summary_path: Path, output_dir: Path) -> None:
    configure_matplotlib()
    summary_rows = read_summary(summary_path)
    history = load_all_histories(summary_rows)
    summary = final_metrics(history)

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "normalized_training_history.csv", index=False)
    summary.to_csv(output_dir / "best_metrics_summary.csv", index=False)
    write_markdown_summary(summary, output_dir / "README.md")

    for metric in METRIC_SPECS:
        plot_metric_curve(history, metric, output_dir)
    plot_metric_grid(history, output_dir)
    plot_final_bars(summary, output_dir)
    plot_training_time(summary, output_dir)
    plot_precision_recall_tradeoff(summary, output_dir)
    plot_rfdetr_class_ap(history, output_dir)

    print(f"Report written to: {output_dir}")
    print(f"Models included: {', '.join(summary['model_name'].astype(str))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate article-style training comparison plots.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/train_compare/summary.jsonl"),
        help="Path to train.py summary.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/train_compare/plots"),
        help="Directory where plots and normalized CSV files will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_report(args.summary.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
