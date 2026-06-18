import argparse
import json
import shutil
import subprocess
from pathlib import Path


RFDETR_CLASSES = {
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
    "large": ("RFDETRLarge", 704),
}


def read_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No successful runs found in {path}")
    return rows


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def resolve_weight_path(row: dict, weights_dir: Path) -> Path:
    original = Path(row["best_weights"])
    if original.exists():
        return original.resolve()
    local = weights_dir / original.name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(f"Could not find weights for {row['run_name']}: {original} or {local}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def export_yolo(row: dict, precision: str, args: argparse.Namespace, manifest_path: Path) -> None:
    from ultralytics import YOLO

    weights = resolve_weight_path(row, args.weights_dir)

    run_dir = args.output_dir / "yolo" / row["run_name"] / precision
    run_dir.mkdir(parents=True, exist_ok=True)
    local_weights = run_dir / weights.name
    if not local_weights.exists() or args.force:
        shutil.copy2(weights, local_weights)

    model = YOLO(str(local_weights))
    common = {
        "format": "engine",
        "imgsz": int(row["input_resolution"]),
        "device": args.device,
        "batch": args.batch,
        "workspace": args.workspace,
        "verbose": True,
    }

    if precision == "fp16":
        export_path = model.export(**common, half=True)
    elif precision == "int8":
        if args.data_yaml is None:
            raise RuntimeError("--data-yaml is required for YOLO INT8 calibration")
        export_path = model.export(**common, int8=True, data=str(args.data_yaml))
    else:
        raise ValueError(f"Unsupported YOLO precision: {precision}")

    export_path = Path(export_path)
    final_engine = run_dir / f"{row['run_name']}_{precision}.engine"
    if export_path.resolve() != final_engine.resolve():
        shutil.copy2(export_path, final_engine)

    append_manifest(
        manifest_path,
        {
            "family": "yolo",
            "runtime": "ultralytics_engine",
            "precision": precision,
            "model": row["model"],
            "run_name": row["run_name"],
            "artifact": str(final_engine.resolve()),
            "input_resolution": row["input_resolution"],
            "dataset_resolution": row["resolution"],
        },
    )
    print(f"YOLO {precision} exported: {final_engine}")


def rfdetr_class(model_key: str):
    import rfdetr

    class_name, _ = RFDETR_CLASSES[model_key]
    return getattr(rfdetr, class_name)


def rfdetr_device(device: str) -> str:
    if device == "cpu" or device.startswith("cuda"):
        return device
    return f"cuda:{device}"


def export_rfdetr_onnx(row: dict, args: argparse.Namespace, manifest_path: Path) -> Path:
    weights = resolve_weight_path(row, args.weights_dir)

    model_key = row["model"]
    model_cls = rfdetr_class(model_key)
    input_resolution = int(row["input_resolution"] or RFDETR_CLASSES[model_key][1])
    run_dir = args.output_dir / "rfdetr" / row["run_name"] / "onnx"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = model_cls(pretrain_weights=str(weights), device=rfdetr_device(args.device))
    model.export(
        output_dir=str(run_dir),
        shape=(input_resolution, input_resolution),
        batch_size=args.batch,
        opset_version=args.opset,
        verbose=False,
    )
    onnx_path = run_dir / "inference_model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"RF-DETR export did not produce {onnx_path}")

    append_manifest(
        manifest_path,
        {
            "family": "rfdetr",
            "runtime": "onnx",
            "precision": "fp32",
            "model": model_key,
            "run_name": row["run_name"],
            "artifact": str(onnx_path.resolve()),
            "source_weights": str(weights),
            "input_resolution": input_resolution,
            "dataset_resolution": row["resolution"],
        },
    )
    print(f"RF-DETR ONNX exported: {onnx_path}")
    return onnx_path


def run_trtexec(onnx_path: Path, engine_path: Path, precision: str, args: argparse.Namespace) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec not found. TensorRT CLI must be available on Jetson.")

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}",
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
    else:
        raise ValueError(precision)

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def export_rfdetr(row: dict, precisions: list[str], args: argparse.Namespace, manifest_path: Path) -> None:
    onnx_path = export_rfdetr_onnx(row, args, manifest_path)
    for precision in precisions:
        if precision == "onnx":
            continue
        if precision == "int8" and not args.rfdetr_int8:
            print(
                f"Skipping RF-DETR INT8 for {row['run_name']}. "
                "Use --rfdetr-int8 only after preparing a proper TensorRT calibration/QDQ flow."
            )
            continue
        engine_dir = args.output_dir / "rfdetr" / row["run_name"] / precision
        engine_dir.mkdir(parents=True, exist_ok=True)
        engine_path = engine_dir / f"{row['run_name']}_{precision}.engine"
        run_trtexec(onnx_path, engine_path, precision, args)
        append_manifest(
            manifest_path,
            {
                "family": "rfdetr",
                "runtime": "tensorrt_engine",
                "precision": precision,
                "model": row["model"],
                "run_name": row["run_name"],
                "artifact": str(engine_path.resolve()),
                "input_resolution": row["input_resolution"],
                "dataset_resolution": row["resolution"],
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained models on Jetson.")
    parser.add_argument("--summary", type=Path, default=Path("runs/train_compare/summary.jsonl"))
    parser.add_argument("--weights-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/exports"))
    parser.add_argument("--data-yaml", type=Path, default=Path("datasets/yolo_640/data.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=int, default=4096, help="TensorRT workspace MiB")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--include", default="yolo,rfdetr")
    parser.add_argument("--yolo-precisions", default="fp16,int8")
    parser.add_argument("--rfdetr-precisions", default="onnx,fp16")
    parser.add_argument("--rfdetr-int8", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reset-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists() and args.reset_manifest:
        manifest_path.unlink()

    include = set(parse_csv(args.include))
    yolo_precisions = parse_csv(args.yolo_precisions)
    rfdetr_precisions = parse_csv(args.rfdetr_precisions)
    rows = read_summary(args.summary)

    for row in rows:
        if row["family"] == "yolo" and "yolo" in include:
            for precision in yolo_precisions:
                export_yolo(row, precision, args, manifest_path)
        elif row["family"] == "rfdetr" and "rfdetr" in include:
            export_rfdetr(row, rfdetr_precisions, args, manifest_path)

    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
