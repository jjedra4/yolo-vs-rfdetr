import argparse
import csv
import json
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


THROUGHPUT_RE = re.compile(r"Throughput:\s+([0-9.]+)\s+qps")
LATENCY_RE = re.compile(
    r"Latency:\s+min\s+=\s+([0-9.]+)\s+ms,\s+max\s+=\s+([0-9.]+)\s+ms,\s+"
    r"mean\s+=\s+([0-9.]+)\s+ms,\s+median\s+=\s+([0-9.]+)\s+ms,\s+"
    r"percentile\(90%\)\s+=\s+([0-9.]+)\s+ms,\s+percentile\(95%\)\s+=\s+([0-9.]+)\s+ms,\s+"
    r"percentile\(99%\)\s+=\s+([0-9.]+)\s+ms"
)
COMPUTE_RE = re.compile(
    r"GPU Compute Time:\s+min\s+=\s+([0-9.]+)\s+ms,\s+max\s+=\s+([0-9.]+)\s+ms,\s+"
    r"mean\s+=\s+([0-9.]+)\s+ms,\s+median\s+=\s+([0-9.]+)\s+ms,\s+"
    r"percentile\(90%\)\s+=\s+([0-9.]+)\s+ms,\s+percentile\(95%\)\s+=\s+([0-9.]+)\s+ms,\s+"
    r"percentile\(99%\)\s+=\s+([0-9.]+)\s+ms"
)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


class TegrastatsCapture:
    def __init__(self, log_path: Path, interval_ms: int):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self.proc = None
        self.file = None

    def __enter__(self):
        if shutil.which("tegrastats") is None:
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
        result[f"tegrastats_{key}_mean"] = sum(values) / len(values)
        result[f"tegrastats_{key}_max"] = max(values)
    return result


def parse_trtexec_output(text: str) -> dict:
    result = {}
    throughput = THROUGHPUT_RE.search(text)
    if throughput:
        result["fps_mean"] = float(throughput.group(1))

    latency = LATENCY_RE.search(text)
    if latency:
        names = [
            "latency_min_ms",
            "latency_max_ms",
            "latency_mean_ms",
            "latency_median_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ]
        result.update({name: float(value) for name, value in zip(names, latency.groups())})

    compute = COMPUTE_RE.search(text)
    if compute:
        names = [
            "gpu_compute_min_ms",
            "gpu_compute_max_ms",
            "gpu_compute_mean_ms",
            "gpu_compute_median_ms",
            "gpu_compute_p90_ms",
            "gpu_compute_p95_ms",
            "gpu_compute_p99_ms",
        ]
        result.update({name: float(value) for name, value in zip(names, compute.groups())})
    return result


def benchmark_engine(row: dict, args: argparse.Namespace) -> dict:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec not found. Add /usr/src/tensorrt/bin to PATH.")

    artifact = Path(row["artifact"])
    if not artifact.exists():
        raise FileNotFoundError(artifact)

    log_stem = f"{row['run_name']}_{row['runtime']}_{row['precision']}"
    stdout_path = args.output_dir / "trtexec_logs" / f"{log_stem}.txt"
    power_path = args.output_dir / "tegrastats" / f"{log_stem}.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        trtexec,
        f"--loadEngine={artifact}",
        f"--warmUp={args.warmup_ms}",
        f"--duration={args.duration_s}",
        "--useCudaGraph",
        "--noDataTransfers",
        "--percentile=90,95,99",
    ]
    print("Running:", " ".join(command))
    with TegrastatsCapture(power_path, args.tegrastats_interval_ms):
        completed = subprocess.run(command, check=False, text=True, capture_output=True)

    text = completed.stdout + "\n" + completed.stderr
    stdout_path.write_text(text, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed for {artifact}. See {stdout_path}")

    result = {
        **row,
        "artifact": str(artifact),
        "benchmark_kind": "raw_trtexec_engine",
        "trtexec_log": str(stdout_path),
        "tegrastats_log": str(power_path),
        "duration_s": args.duration_s,
        "warmup_ms": args.warmup_ms,
    }
    result.update(parse_trtexec_output(text))
    result.update(parse_tegrastats(power_path))
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engines with trtexec.")
    parser.add_argument("--manifest", type=Path, default=Path("deploy/exports/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("jetson_results/trtexec_engines"))
    parser.add_argument("--families", default="yolo,rfdetr")
    parser.add_argument("--runtimes", default="ultralytics_engine,tensorrt_engine")
    parser.add_argument("--precisions", default="fp16,int8")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--duration-s", type=int, default=10)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = parse_csv(args.families)
    runtimes = parse_csv(args.runtimes)
    precisions = parse_csv(args.precisions)

    rows = [
        row
        for row in read_manifest(args.manifest)
        if row.get("family") in families
        and row.get("runtime") in runtimes
        and row.get("precision") in precisions
        and Path(row.get("artifact", "")).suffix == ".engine"
    ]
    if not rows:
        raise RuntimeError("No engine rows selected from manifest")

    results = []
    for row in rows:
        print(f"Benchmarking {row['run_name']} [{row['runtime']} {row['precision']}]")
        results.append(benchmark_engine(row, args))
        write_csv(results, args.output_dir / "trtexec_engine_summary.csv")

    print(f"Summary: {args.output_dir / 'trtexec_engine_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
