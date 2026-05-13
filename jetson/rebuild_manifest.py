import argparse
import json
from pathlib import Path


def read_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows.append(row)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def rebuild(summary_path: Path, exports_dir: Path) -> list[dict]:
    manifest = []
    for row in read_summary(summary_path):
        run_name = row["run_name"]
        family = row["family"]
        model = row["model"]
        input_resolution = row.get("input_resolution")
        dataset_resolution = row.get("resolution")

        if family == "yolo":
            for precision in ["fp16", "int8"]:
                base = exports_dir / "yolo" / run_name / precision
                artifact = find_first(
                    [
                        base / f"{run_name}_{precision}.engine",
                        base / f"{run_name}_best.engine",
                    ]
                )
                if artifact:
                    manifest.append(
                        {
                            "family": "yolo",
                            "runtime": "ultralytics_engine",
                            "precision": precision,
                            "model": model,
                            "run_name": run_name,
                            "artifact": str(artifact.resolve()),
                            "input_resolution": input_resolution,
                            "dataset_resolution": dataset_resolution,
                        }
                    )

        elif family == "rfdetr":
            onnx_path = exports_dir / "rfdetr" / run_name / "onnx" / "inference_model.onnx"
            if onnx_path.exists():
                manifest.append(
                    {
                        "family": "rfdetr",
                        "runtime": "onnx",
                        "precision": "fp32",
                        "model": model,
                        "run_name": run_name,
                        "artifact": str(onnx_path.resolve()),
                        "input_resolution": input_resolution,
                        "dataset_resolution": dataset_resolution,
                    }
                )

            for precision in ["fp16", "int8"]:
                engine_path = exports_dir / "rfdetr" / run_name / precision / f"{run_name}_{precision}.engine"
                if engine_path.exists():
                    manifest.append(
                        {
                            "family": "rfdetr",
                            "runtime": "tensorrt_engine",
                            "precision": precision,
                            "model": model,
                            "run_name": run_name,
                            "artifact": str(engine_path.resolve()),
                            "input_resolution": input_resolution,
                            "dataset_resolution": dataset_resolution,
                        }
                    )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild deploy/exports/manifest.jsonl from existing export files.")
    parser.add_argument("--summary", type=Path, default=Path("runs/train_compare/summary.jsonl"))
    parser.add_argument("--exports-dir", type=Path, default=Path("deploy/exports"))
    parser.add_argument("--output", type=Path, default=Path("deploy/exports/manifest.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = rebuild(args.summary, args.exports_dir)
    if not rows:
        raise RuntimeError(f"No exported artifacts found under {args.exports_dir}")
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    for row in rows:
        print(f"{row['family']} {row['run_name']} {row['runtime']} {row['precision']} -> {row['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
