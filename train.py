import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from ultralytics import YOLO


load_dotenv()


ROBOFLOW_WORKSPACE = "objectdetection-z2zn1"
ROBOFLOW_PROJECT = "garbage-classification-3-bjcgp"
ROBOFLOW_VERSION_BY_RESOLUTION = {
    416: 3,
    640: 4,
    1280: 5,
}

DEFAULT_YOLO_MODELS = ["yolov8s.pt", "yolov8m.pt", "yolo11s.pt", "yolo11m.pt"]
DEFAULT_RFDETR_MODELS = ["small", "medium"]
DEFAULT_RESOLUTIONS = [640]

RFDETR_MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

RFDETR_MODEL_DIVISORS = {
    "nano": 32,
    "small": 32,
    "medium": 32,
    "base": 56,
    "large": 32,
}

RFDETR_DEFAULT_RESOLUTIONS = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "base": 560,
    "large": 704,
}


@dataclass(frozen=True)
class DatasetPaths:
    resolution: int
    root: Path
    yolo_yaml: Path


@dataclass
class RunRecord:
    status: str
    family: str
    model: str
    resolution: int
    run_name: str
    output_dir: str
    input_resolution: int | None = None
    best_weights: str | None = None
    seconds: float | None = None
    error: str | None = None


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_auto_int(value: str) -> int | str:
    if value == "auto":
        return value
    return int(value)


def parse_optional_int(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def slug_model_name(model_name: str) -> str:
    return Path(model_name).stem.replace("-", "_").replace(".", "_")


def resolve_device(requested_device: str, allow_cpu: bool) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        requested_device = "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see CUDA. "
            "Install a CUDA PyTorch build and check that the NVIDIA driver works."
        )

    if requested_device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Training resolved to CPU. Use a CUDA device for the real run, "
            "or pass --allow-cpu only for debugging. "
            f"Current torch build: {torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "If this is a CPU-only build, reinstall with: "
            "python -m pip uninstall -y torch torchvision && "
            "python -m pip install --no-cache-dir --force-reinstall -r requirements-gpu-cu128.txt"
        )

    return requested_device


def print_device_summary(device: str) -> None:
    print("=" * 80)
    print("Device")
    print("=" * 80)
    print(f"requested/resolved: {device}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            memory_gb = props.total_memory / 1024**3
            print(f"cuda:{idx}: {props.name} ({memory_gb:.1f} GB)")
    print()


def validate_resolution(resolution: int) -> None:
    if resolution not in ROBOFLOW_VERSION_BY_RESOLUTION:
        allowed = ", ".join(str(item) for item in sorted(ROBOFLOW_VERSION_BY_RESOLUTION))
        raise ValueError(f"Unsupported resolution {resolution}. Known Roboflow versions: {allowed}.")


def validate_rfdetr_resolution(model_key: str, resolution: int) -> None:
    divisor = RFDETR_MODEL_DIVISORS[model_key]
    if resolution % divisor != 0:
        raise ValueError(
            f"RF-DETR {model_key} cannot use input resolution {resolution}. "
            f"It must be divisible by {divisor}. Use a compatible resolution or remove this model."
        )


def rfdetr_default_resolution(model_key: str) -> int:
    return RFDETR_DEFAULT_RESOLUTIONS[model_key]


def resolve_yolo_imgsz_values(value: str, rfdetr_models: list[str], rfdetr_input_resolution: int | None) -> list[int | None]:
    if value == "dataset":
        return [None]
    if value == "match-rfdetr":
        if not rfdetr_models and rfdetr_input_resolution is None:
            raise ValueError("--yolo-imgsz match-rfdetr needs at least one RF-DETR model or --rfdetr-input-resolution.")
        if rfdetr_input_resolution:
            return [rfdetr_input_resolution]
        return sorted({rfdetr_default_resolution(model_key) for model_key in rfdetr_models})
    return parse_int_csv(value)


def dataset_paths(data_root: Path, resolution: int) -> DatasetPaths:
    root = data_root / f"yolo_{resolution}"
    return DatasetPaths(
        resolution=resolution,
        root=root,
        yolo_yaml=root / "data.yaml",
    )


def validate_dataset(paths: DatasetPaths) -> None:
    required = [
        paths.yolo_yaml,
        paths.root / "train" / "images",
        paths.root / "train" / "labels",
        paths.root / "valid" / "images",
        paths.root / "valid" / "labels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset for {paths.resolution}px is incomplete in {paths.root}. Missing: {missing}"
        )


def download_dataset_if_needed(
    data_root: Path,
    resolution: int,
    mode: str,
    roboflow_api_key: str | None,
) -> DatasetPaths:
    validate_resolution(resolution)
    paths = dataset_paths(data_root, resolution)

    if mode != "force":
        try:
            validate_dataset(paths)
            print(f"Dataset {resolution}px OK: {paths.root}")
            return paths
        except FileNotFoundError:
            if mode == "never":
                raise

    if not roboflow_api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is required to download missing datasets. "
            "Put it in .env or pass --download never when datasets already exist."
        )

    print(f"Downloading Roboflow YOLOv8 dataset for {resolution}px to {paths.root}")
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION_BY_RESOLUTION[resolution])
    downloaded = version.download(
        "yolov8",
        location=str(paths.root),
        overwrite=(mode == "force"),
    )

    final_paths = DatasetPaths(
        resolution=resolution,
        root=Path(downloaded.location),
        yolo_yaml=Path(downloaded.location) / "data.yaml",
    )
    validate_dataset(final_paths)
    return final_paths


def make_comet_experiment(
    enabled: bool,
    strict: bool,
    project_name: str,
    run_name: str,
    params: dict[str, Any],
):
    if not enabled:
        return None

    from comet_ml import Experiment

    api_key = os.getenv("COMET_API_KEY")
    try:
        experiment = Experiment(api_key=api_key, project_name=project_name, auto_output_logging="simple")
        experiment.set_name(run_name)
        experiment.log_parameters(params)
        return experiment
    except Exception as exc:
        if strict:
            raise
        print(f"Comet logging disabled for {run_name}: {exc!r}")
        return None


def log_metrics_csv_to_comet(experiment: Any, csv_path: Path) -> None:
    if experiment is None or not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "epoch" not in df.columns:
        return

    for _, row in df.iterrows():
        epoch = int(row["epoch"])
        for col, value in row.items():
            if col == "epoch" or pd.isna(value):
                continue
            try:
                experiment.log_metric(col, float(value), step=epoch)
            except (TypeError, ValueError):
                continue


def append_run_record(summary_path: Path, record: RunRecord) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def maybe_copy_best_weight(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def train_yolo_run(
    model_name: str,
    dataset: DatasetPaths,
    input_resolution: int,
    args: argparse.Namespace,
    device: str,
    comet_enabled: bool,
    summary_path: Path,
) -> RunRecord:
    run_name = f"{slug_model_name(model_name)}_data{dataset.resolution}_img{input_resolution}"
    output_dir = args.project_dir / "yolo" / run_name
    last_weights = output_dir / "weights" / "last.pt"
    best_weights = output_dir / "weights" / "best.pt"
    exported_best = args.weights_dir / f"{run_name}_best.pt"
    started = time.perf_counter()
    experiment = None

    print("=" * 80)
    print(f"YOLO start: {run_name}")
    print("=" * 80)

    try:
        params = {
            "family": "yolo",
            "model": model_name,
            "dataset_resolution": dataset.resolution,
            "input_resolution": input_resolution,
            "epochs": args.epochs,
            "batch": args.yolo_batch,
            "device": device,
            "dataset": str(dataset.root),
            "seed": args.seed,
        }
        experiment = make_comet_experiment(
            comet_enabled,
            args.comet == "on",
            args.comet_project,
            run_name,
            params,
        )

        if args.resume and last_weights.exists():
            model = YOLO(str(last_weights))
            model.train(resume=True)
        else:
            model = YOLO(model_name)
            model.train(
                data=str(dataset.yolo_yaml),
                epochs=args.epochs,
                imgsz=input_resolution,
                batch=args.yolo_batch,
                device=device,
                project=str(args.project_dir / "yolo"),
                name=run_name,
                exist_ok=args.exist_ok,
                plots=True,
                val=True,
                seed=args.seed,
                workers=args.workers,
                patience=args.patience,
                cos_lr=True,
                cache=args.cache,
            )

        log_metrics_csv_to_comet(experiment, output_dir / "results.csv")
        copied = maybe_copy_best_weight(best_weights, exported_best)
        record = RunRecord(
            status="ok",
            family="yolo",
            model=model_name,
            resolution=dataset.resolution,
            run_name=run_name,
            output_dir=str(output_dir),
            input_resolution=input_resolution,
            best_weights=copied or (str(best_weights) if best_weights.exists() else None),
            seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        record = RunRecord(
            status="failed",
            family="yolo",
            model=model_name,
            resolution=dataset.resolution,
            run_name=run_name,
            output_dir=str(output_dir),
            input_resolution=input_resolution,
            seconds=time.perf_counter() - started,
            error=repr(exc),
        )
        if not args.continue_on_error:
            append_run_record(summary_path, record)
            raise
    finally:
        if experiment is not None:
            experiment.end()

    append_run_record(summary_path, record)
    return record


def rfdetr_best_checkpoint(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "checkpoint_best_total.pth",
        output_dir / "checkpoint_best_regular.pth",
        output_dir / "checkpoint.pth",
        output_dir / "last.pth",
    ]
    return next((path for path in candidates if path.exists()), None)


def rfdetr_metrics_csv(output_dir: Path) -> Path:
    for filename in ["metrics.csv", "results.csv"]:
        candidate = output_dir / filename
        if candidate.exists():
            return candidate
    return output_dir / "metrics.csv"


def train_rfdetr_run(
    model_key: str,
    dataset: DatasetPaths,
    args: argparse.Namespace,
    device: str,
    comet_enabled: bool,
    summary_path: Path,
) -> RunRecord:
    input_resolution = args.rfdetr_input_resolution or rfdetr_default_resolution(model_key)
    run_name = f"rfdetr_{model_key}_data{dataset.resolution}_img{input_resolution}"
    output_dir = args.project_dir / "rfdetr" / run_name
    resume_checkpoint = output_dir / "checkpoint.pth"
    exported_best = args.weights_dir / f"{run_name}_best.pth"
    started = time.perf_counter()
    experiment = None

    print("=" * 80)
    print(f"RF-DETR start: {run_name}")
    print("=" * 80)

    try:
        model_cls = RFDETR_MODEL_CLASSES[model_key]
        params = {
            "family": "rfdetr",
            "model": model_key,
            "dataset_resolution": dataset.resolution,
            "input_resolution": input_resolution,
            "epochs": args.epochs,
            "batch_size": args.rfdetr_batch,
            "grad_accum_steps": args.rfdetr_grad_accum_steps,
            "device": device,
            "dataset": str(dataset.root),
            "seed": args.seed,
        }
        experiment = make_comet_experiment(
            comet_enabled,
            args.comet == "on",
            args.comet_project,
            run_name,
            params,
        )

        model = model_cls(device=device)
        model.train(
            dataset_dir=str(dataset.root),
            dataset_file="roboflow",
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.rfdetr_batch,
            grad_accum_steps=args.rfdetr_grad_accum_steps,
            **({"resolution": args.rfdetr_input_resolution} if args.rfdetr_input_resolution else {}),
            device=device,
            seed=args.seed,
            num_workers=args.workers,
            resume=str(resume_checkpoint) if args.resume and resume_checkpoint.exists() else None,
            tensorboard=True,
            wandb=False,
            mlflow=False,
            run_test=False,
            progress_bar="tqdm",
        )

        metrics_csv = rfdetr_metrics_csv(output_dir)
        log_metrics_csv_to_comet(experiment, metrics_csv)

        best_checkpoint = rfdetr_best_checkpoint(output_dir)
        copied = maybe_copy_best_weight(best_checkpoint, exported_best) if best_checkpoint else None
        record = RunRecord(
            status="ok",
            family="rfdetr",
            model=model_key,
            resolution=dataset.resolution,
            run_name=run_name,
            output_dir=str(output_dir),
            input_resolution=input_resolution,
            best_weights=copied or (str(best_checkpoint) if best_checkpoint else None),
            seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        record = RunRecord(
            status="failed",
            family="rfdetr",
            model=model_key,
            resolution=dataset.resolution,
            run_name=run_name,
            output_dir=str(output_dir),
            input_resolution=input_resolution,
            seconds=time.perf_counter() - started,
            error=repr(exc),
        )
        if not args.continue_on_error:
            append_run_record(summary_path, record)
            raise
    finally:
        if experiment is not None:
            experiment.end()

    append_run_record(summary_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO and RF-DETR models on the Roboflow garbage dataset."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS)))
    parser.add_argument("--yolo-models", default=",".join(DEFAULT_YOLO_MODELS))
    parser.add_argument(
        "--yolo-imgsz",
        default="dataset",
        help=(
            "YOLO square input size list, e.g. 512,576. Use dataset to match each "
            "dataset version, or match-rfdetr to use RF-DETR model defaults."
        ),
    )
    parser.add_argument("--rfdetr-models", default=",".join(DEFAULT_RFDETR_MODELS))
    parser.add_argument(
        "--rfdetr-input-resolution",
        default="auto",
        type=parse_optional_int,
        help=(
            "RF-DETR square input size. Use auto to keep each pretrained model default "
            "(small=512, medium=576). Do not set this to dataset resolution unless you "
            "know the checkpoint supports it."
        ),
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-rfdetr", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--yolo-batch", default=0.8, type=float)
    parser.add_argument("--rfdetr-batch", default=4, type=parse_auto_int)
    parser.add_argument("--rfdetr-grad-accum-steps", type=int, default=4)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--download", choices=["missing", "force", "never"], default="missing")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--project-dir", type=Path, default=Path("runs") / "train_compare")
    parser.add_argument("--weights-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--comet", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--comet-project", default=os.getenv("COMET_PROJECT", "garbage-classification-comparison"))
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(exist_ok=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()

    resolutions = parse_int_csv(args.resolutions)
    yolo_models = [] if args.skip_yolo else parse_csv(args.yolo_models)
    rfdetr_models = [] if args.skip_rfdetr else parse_csv(args.rfdetr_models)
    yolo_imgsz_values = [] if args.skip_yolo else resolve_yolo_imgsz_values(
        args.yolo_imgsz,
        rfdetr_models,
        args.rfdetr_input_resolution,
    )

    for resolution in resolutions:
        validate_resolution(resolution)
    unknown_rfdetr = [model for model in rfdetr_models if model not in RFDETR_MODEL_CLASSES]
    if unknown_rfdetr:
        allowed = ", ".join(sorted(RFDETR_MODEL_CLASSES))
        raise ValueError(f"Unknown RF-DETR model(s): {unknown_rfdetr}. Allowed: {allowed}")
    for resolution in resolutions:
        for model_name in rfdetr_models:
            if args.rfdetr_input_resolution:
                validate_rfdetr_resolution(model_name, args.rfdetr_input_resolution)

    device = resolve_device(args.device, args.allow_cpu)
    print_device_summary(device)

    comet_api_key = os.getenv("COMET_API_KEY")
    comet_enabled = args.comet == "on" or (args.comet == "auto" and bool(comet_api_key))
    if args.comet == "on" and not comet_api_key:
        raise RuntimeError("COMET_API_KEY is required when --comet on is used.")
    print(f"Comet logging: {'on' if comet_enabled else 'off'}")

    queue = []
    for resolution in resolutions:
        for model_name in yolo_models:
            for yolo_imgsz in yolo_imgsz_values:
                input_resolution = resolution if yolo_imgsz is None else yolo_imgsz
                queue.append(("yolo", model_name, resolution, input_resolution))
        for model_name in rfdetr_models:
            input_resolution = args.rfdetr_input_resolution or rfdetr_default_resolution(model_name)
            queue.append(("rfdetr", model_name, resolution, input_resolution))

    print("=" * 80)
    print("Training queue")
    print("=" * 80)
    for family, model_name, resolution, input_resolution in queue:
        print(f"{family:7s} {model_name:14s} data={resolution}px input={input_resolution}px")
    print(f"Total runs: {len(queue)}")
    print()

    if args.dry_run:
        return 0

    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    datasets = {
        resolution: download_dataset_if_needed(args.data_root, resolution, args.download, roboflow_api_key)
        for resolution in resolutions
    }

    args.project_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.project_dir / "summary.jsonl"
    if summary_path.exists() and not args.resume:
        summary_path.unlink()

    for family, model_name, resolution, input_resolution in queue:
        dataset = datasets[resolution]
        if family == "yolo":
            train_yolo_run(model_name, dataset, input_resolution, args, device, comet_enabled, summary_path)
        elif family == "rfdetr":
            train_rfdetr_run(model_name, dataset, args, device, comet_enabled, summary_path)
        else:
            raise AssertionError(f"Unexpected family: {family}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Summary: {summary_path}")
    print(f"Best weights copied to: {args.weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
