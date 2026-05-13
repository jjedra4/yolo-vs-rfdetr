import argparse
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "YOLOv8-S",
    "YOLOv8-M",
    "YOLO11-S",
    "YOLO11-M",
    "RF-DETR-S",
    "RF-DETR-M",
]

PRECISION_ORDER = ["fp32", "fp16", "int8"]
BENCHMARK_ORDER = ["end_to_end", "raw_engine"]


def model_label(row: pd.Series) -> str:
    family = str(row.get("family", ""))
    model = str(row.get("model", ""))
    run_name = str(row.get("run_name", ""))

    if family == "yolo":
        key = Path(model).stem.lower()
        if not key or key == "nan":
            key = run_name.split("_data", 1)[0].lower()
        return {
            "yolov8s": "YOLOv8-S",
            "yolov8m": "YOLOv8-M",
            "yolo11s": "YOLO11-S",
            "yolo11m": "YOLO11-M",
        }.get(key, key.upper())

    if family == "rfdetr":
        key = model.lower()
        if not key or key == "nan":
            key = run_name.replace("rfdetr_", "").split("_", 1)[0]
        return {
            "small": "RF-DETR-S",
            "medium": "RF-DETR-M",
            "large": "RF-DETR-L",
        }.get(key, f"RF-DETR-{key.upper()}")

    return run_name


def normalize_precision(row: pd.Series) -> str:
    precision = str(row.get("precision", "")).lower()
    runtime = str(row.get("runtime", ""))
    if precision and precision != "nan":
        return precision
    if runtime.endswith("_pt") or runtime == "rfdetr_pytorch":
        return "fp32"
    return "unknown"


def benchmark_kind(row: pd.Series, source_kind: str) -> str:
    if source_kind == "raw_engine":
        return "raw_engine"
    return "end_to_end"


def read_csv_if_exists(path: Path, source_kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["source_file"] = str(path)
    df["benchmark_kind"] = source_kind
    return df


def best_power_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "tegrastats_VDD_IN_avg_mw_mean",
        "tegrastats_VDD_CPU_GPU_CV_avg_mw_mean",
        "tegrastats_VDD_GPU_SOC_avg_mw_mean",
        "tegrastats_VDD_GPU_avg_mw_mean",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    candidates = [col for col in df.columns if col.startswith("tegrastats_") and col.endswith("_avg_mw_mean")]
    return candidates[0] if candidates else None


def normalize_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if not df.empty]
    if not frames:
        raise RuntimeError("No benchmark CSV files found.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["model_label"] = df.apply(model_label, axis=1)
    df["precision"] = df.apply(normalize_precision, axis=1)

    power_col = best_power_column(df)
    df["power_mean_w"] = pd.NA
    if power_col:
        df["power_mean_w"] = pd.to_numeric(df[power_col], errors="coerce") / 1000.0

    if "latency_median_ms" not in df.columns and "latency_mean_ms" in df.columns:
        df["latency_median_ms"] = pd.NA
    if "latency_p95_ms" not in df.columns:
        df["latency_p95_ms"] = pd.NA
    if "latency_p99_ms" not in df.columns:
        df["latency_p99_ms"] = pd.NA

    df["fps_mean"] = pd.to_numeric(df["fps_mean"], errors="coerce")
    df["latency_mean_ms"] = pd.to_numeric(df["latency_mean_ms"], errors="coerce")
    df["latency_median_ms"] = pd.to_numeric(df["latency_median_ms"], errors="coerce")
    df["latency_p95_ms"] = pd.to_numeric(df["latency_p95_ms"], errors="coerce")
    df["latency_p99_ms"] = pd.to_numeric(df["latency_p99_ms"], errors="coerce")

    df["energy_per_frame_j"] = pd.NA
    mask = df["power_mean_w"].notna() & df["fps_mean"].notna() & (df["fps_mean"] > 0)
    df.loc[mask, "energy_per_frame_j"] = df.loc[mask, "power_mean_w"] / df.loc[mask, "fps_mean"]

    keep = [
        "model_label",
        "family",
        "precision",
        "benchmark_kind",
        "runtime",
        "run_name",
        "input_resolution",
        "dataset_resolution",
        "fps_mean",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "power_mean_w",
        "energy_per_frame_j",
        "artifact",
        "source_file",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[keep].copy()
    out["model_label"] = pd.Categorical(out["model_label"], MODEL_ORDER, ordered=True)
    out["precision"] = pd.Categorical(out["precision"], PRECISION_ORDER, ordered=True)
    out["benchmark_kind"] = pd.Categorical(out["benchmark_kind"], BENCHMARK_ORDER, ordered=True)
    return out.sort_values(["model_label", "precision", "benchmark_kind", "runtime"]).reset_index(drop=True)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "model_label",
        "precision",
        "benchmark_kind",
        "runtime",
        "input_resolution",
        "fps_mean",
        "latency_mean_ms",
        "latency_p95_ms",
        "power_mean_w",
        "energy_per_frame_j",
    ]
    table = df[display_cols].copy()
    for col in ["fps_mean", "latency_mean_ms", "latency_p95_ms", "power_mean_w", "energy_per_frame_j"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    header = "| " + " | ".join(display_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.astype(str).itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([header, separator, *rows]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Jetson benchmark CSVs into one paper-ready table.")
    parser.add_argument("--pytorch", type=Path, default=Path("jetson_results/pytorch_baseline/benchmark_summary.csv"))
    parser.add_argument("--yolo-trt", type=Path, default=Path("jetson_results/yolo_tensorrt/benchmark_summary.csv"))
    parser.add_argument("--trtexec", type=Path, default=Path("jetson_results/trtexec_engines/trtexec_engine_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("jetson_results/tables"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = normalize_table(
        [
            read_csv_if_exists(args.pytorch, "end_to_end"),
            read_csv_if_exists(args.yolo_trt, "end_to_end"),
            read_csv_if_exists(args.trtexec, "raw_engine"),
        ]
    )

    csv_path = args.output_dir / "jetson_inference_table.csv"
    md_path = args.output_dir / "jetson_inference_table.md"
    table.to_csv(csv_path, index=False)
    write_markdown(table, md_path)

    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(table[["model_label", "precision", "benchmark_kind", "runtime", "fps_mean", "latency_mean_ms"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
