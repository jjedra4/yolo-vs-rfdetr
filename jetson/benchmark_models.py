import argparse
import csv
import json
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image


RFDETR_CLASSES = {
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_weight_path(best_weights: str, weights_dir: Path) -> str:
    original = Path(best_weights)
    if original.exists():
        return str(original.resolve())
    local = weights_dir / original.name
    if local.exists():
        return str(local.resolve())
    return str(original)


def rows_from_summary(path: Path, weights_dir: Path) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        if row["family"] == "yolo":
            rows.append(
                {
                    "family": "yolo",
                    "runtime": "ultralytics_pt",
                    "precision": "fp32",
                    "model": row["model"],
                    "run_name": row["run_name"],
                    "artifact": resolve_weight_path(row["best_weights"], weights_dir),
                    "input_resolution": row["input_resolution"],
                    "dataset_resolution": row["resolution"],
                }
            )
        elif row["family"] == "rfdetr":
            rows.append(
                {
                    "family": "rfdetr",
                    "runtime": "rfdetr_pytorch",
                    "precision": "fp32",
                    "model": row["model"],
                    "run_name": row["run_name"],
                    "artifact": resolve_weight_path(row["best_weights"], weights_dir),
                    "input_resolution": row["input_resolution"],
                    "dataset_resolution": row["resolution"],
                }
            )
    return rows


def list_images(images_dir: Path, limit: int | None) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in suffixes)
    if limit:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")
    return images


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TegrastatsCapture:
    def __init__(self, log_path: Path, interval_ms: int = 250):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self.proc = None
        self.file = None

    def __enter__(self):
        if shutil_which("tegrastats") is None:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=self.file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.5)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.file is not None:
            self.file.close()


def shutil_which(command: str):
    from shutil import which

    return which(command)


def parse_tegrastats(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rails = {}
    for rail, current, average in re.findall(r"([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW", text):
        rails.setdefault(f"{rail}_current_mw", []).append(float(current))
        rails.setdefault(f"{rail}_avg_mw", []).append(float(average))

    result = {}
    for key, values in rails.items():
        if values:
            result[f"tegrastats_{key}_mean"] = statistics.fmean(values)
            result[f"tegrastats_{key}_max"] = max(values)
    return result


def load_yolo_model(row: dict):
    from ultralytics import YOLO

    return YOLO(row["artifact"])


def predict_yolo(model, image_path: Path, row: dict, args: argparse.Namespace):
    return model.predict(
        str(image_path),
        imgsz=int(row["input_resolution"]),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )


def load_rfdetr_model(row: dict, args: argparse.Namespace):
    import rfdetr

    model_cls = getattr(rfdetr, RFDETR_CLASSES[row["model"]])
    return model_cls(pretrain_weights=row["artifact"], device=f"cuda:{args.device}" if args.device != "cpu" else "cpu")


def predict_rfdetr(model, image_path: Path, row: dict, args: argparse.Namespace):
    image = Image.open(image_path).convert("RGB")
    return model.predict(image, threshold=args.conf, include_source_image=False)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def benchmark_one(row: dict, images: list[Path], args: argparse.Namespace, output_dir: Path) -> dict:
    runtime = row["runtime"]
    if runtime in {"ultralytics_pt", "ultralytics_engine"}:
        model = load_yolo_model(row)
        predict = lambda image_path: predict_yolo(model, image_path, row, args)
    elif runtime == "rfdetr_pytorch":
        model = load_rfdetr_model(row, args)
        predict = lambda image_path: predict_rfdetr(model, image_path, row, args)
    else:
        raise ValueError(f"Unsupported runtime for benchmark: {runtime}")

    warmup_images = images[: max(1, min(args.warmup, len(images)))]
    for image_path in warmup_images:
        predict(image_path)
    cuda_sync()

    latencies_ms = []
    raw_rows = []
    power_log = output_dir / "tegrastats" / f"{row['run_name']}_{runtime}_{row['precision']}.log"
    with TegrastatsCapture(power_log, interval_ms=args.tegrastats_interval_ms):
        for repeat in range(args.repeats):
            for image_path in images:
                cuda_sync()
                start = time.perf_counter()
                predict(image_path)
                cuda_sync()
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)
                raw_rows.append(
                    {
                        "run_name": row["run_name"],
                        "runtime": runtime,
                        "precision": row["precision"],
                        "repeat": repeat,
                        "image": str(image_path),
                        "latency_ms": elapsed_ms,
                    }
                )

    raw_path = output_dir / "raw_latency.csv"
    write_header = not raw_path.exists()
    with raw_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(raw_rows)

    mean_latency = statistics.fmean(latencies_ms)
    summary = {
        **row,
        "images": len(images),
        "repeats": args.repeats,
        "samples": len(latencies_ms),
        "latency_mean_ms": mean_latency,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_p99_ms": percentile(latencies_ms, 99),
        "fps_mean": 1000.0 / mean_latency if mean_latency > 0 else 0,
        "tegrastats_log": str(power_log),
    }
    summary.update(parse_tegrastats(power_log))
    return summary


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trained models on Jetson.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", type=Path, help="train.py summary.jsonl for PyTorch baselines")
    source.add_argument("--manifest", type=Path, help="deploy/export manifest for exported engines")
    parser.add_argument("--weights-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--images", type=Path, default=Path("datasets/yolo_640/valid/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("jetson_results"))
    parser.add_argument("--include-runtime", default="ultralytics_pt,ultralytics_engine,rfdetr_pytorch")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(args.images, args.limit)
    rows = rows_from_summary(args.summary, args.weights_dir) if args.summary else read_jsonl(args.manifest)
    include_runtime = set(parse_csv(args.include_runtime))
    rows = [row for row in rows if row.get("runtime") in include_runtime]
    if not rows:
        raise RuntimeError("No models selected for benchmark")

    if args.dry_run:
        print(f"Images: {len(images)} from {args.images}")
        for row in rows:
            print(f"{row['run_name']} [{row['runtime']} {row['precision']}] -> {row['artifact']}")
        return 0

    summaries = []
    for row in rows:
        print(f"Benchmarking {row['run_name']} [{row['runtime']} {row['precision']}]")
        summaries.append(benchmark_one(row, images, args, args.output_dir))
        write_summary_csv(summaries, args.output_dir / "benchmark_summary.csv")

    print(f"Summary: {args.output_dir / 'benchmark_summary.csv'}")
    print(f"Raw latency: {args.output_dir / 'raw_latency.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
