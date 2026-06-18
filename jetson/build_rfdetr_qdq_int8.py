import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def read_manifest(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def list_images(images_dir: Path, max_images: int, seed: int) -> list[Path]:
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No calibration images found in {images_dir}")
    if max_images and len(images) > max_images:
        rng = random.Random(seed)
        images = rng.sample(images, max_images)
        images.sort()
    return images


def preprocess_image(path: Path, height: int, width: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


def onnx_input_info(path: Path, fallback_resolution: int, batch: int) -> tuple[str, tuple[int, int, int, int]]:
    model = onnx.load(str(path), load_external_data=False)
    if len(model.graph.input) != 1:
        raise RuntimeError(f"Expected one RF-DETR ONNX input, got {len(model.graph.input)}")

    inp = model.graph.input[0]
    dims = []
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    if len(dims) != 4:
        raise RuntimeError(f"Expected NCHW RF-DETR ONNX input, got {inp.name}: {dims}")

    dims[0] = batch if dims[0] < 0 else dims[0]
    dims[1] = 3 if dims[1] < 0 else dims[1]
    dims[2] = fallback_resolution if dims[2] < 0 else dims[2]
    dims[3] = fallback_resolution if dims[3] < 0 else dims[3]
    if dims[0] != batch:
        raise RuntimeError(f"ONNX has static batch {dims[0]}, but --batch={batch}")
    return inp.name, tuple(dims)


class RfdetrCalibrationReader:
    def __init__(
        self,
        image_paths: list[Path],
        input_name: str,
        input_shape: tuple[int, int, int, int],
    ) -> None:
        self.image_paths = image_paths
        self.input_name = input_name
        self.batch, self.channels, self.height, self.width = input_shape
        if self.batch != 1:
            raise RuntimeError("This Q/DQ calibration reader currently supports --batch 1 only.")
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        image = preprocess_image(self.image_paths[self.index], self.height, self.width)
        self.index += 1
        return {self.input_name: image[None, ...].astype(np.float32)}


def selected_onnx_rows(manifest_rows: list[dict], models: set[str]) -> list[dict]:
    rows = []
    for row in manifest_rows:
        if row.get("family") != "rfdetr":
            continue
        if row.get("runtime") != "onnx":
            continue
        if row.get("precision") != "fp32":
            continue
        if models and row.get("model") not in models:
            continue
        rows.append(row)
    return rows


def quantize_qdq(
    onnx_path: Path,
    qdq_path: Path,
    calibration_images: list[Path],
    input_name: str,
    input_shape: tuple[int, int, int, int],
    per_channel: bool,
) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    reader = RfdetrCalibrationReader(calibration_images, input_name, input_shape)
    qdq_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(qdq_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=per_channel,
        op_types_to_quantize=["Conv", "MatMul"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    fold_dequantized_bias_initializers(qdq_path)


def fold_dequantized_bias_initializers(model_path: Path) -> None:
    model = onnx.load(str(model_path))
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    folded_outputs = set()
    kept_nodes = []

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 2 or len(node.output) != 1:
            kept_nodes.append(node)
            continue

        output_name = node.output[0]
        quantized_name = node.input[0]
        scale_name = node.input[1]
        zero_name = node.input[2] if len(node.input) >= 3 else None
        if not output_name.endswith(".bias") and "bias_quantized" not in quantized_name:
            kept_nodes.append(node)
            continue
        if quantized_name not in initializers or scale_name not in initializers:
            kept_nodes.append(node)
            continue

        quantized = numpy_helper.to_array(initializers[quantized_name])
        scale = numpy_helper.to_array(initializers[scale_name])
        zero_point = np.array(0, dtype=quantized.dtype)
        if zero_name and zero_name in initializers:
            zero_point = numpy_helper.to_array(initializers[zero_name])

        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
                break

        if scale.ndim > 0 and scale.size > 1 and quantized.ndim > 1:
            shape = [1] * quantized.ndim
            shape[axis] = scale.shape[0]
            scale = scale.reshape(shape)
            zero_point = zero_point.reshape(shape)

        folded = (quantized.astype(np.float32) - zero_point.astype(np.float32)) * scale.astype(np.float32)
        model.graph.initializer.append(numpy_helper.from_array(folded.astype(np.float32), output_name))
        folded_outputs.add(output_name)

    if not folded_outputs:
        return

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    onnx.save(model, str(model_path))
    print(f"Folded {len(folded_outputs)} quantized bias tensors back to FP32 for TensorRT compatibility.")


def run_trtexec(qdq_path: Path, engine_path: Path, workspace_mib: int, use_int8_flag: bool) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec not found. Add /usr/src/tensorrt/bin to PATH.")

    command = [
        trtexec,
        f"--onnx={qdq_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    if use_int8_flag:
        command.append("--int8")
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def manifest_has_artifact(path: Path, artifact: Path) -> bool:
    artifact_text = str(artifact.resolve())
    return any(row.get("artifact") == artifact_text for row in read_manifest(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RF-DETR TensorRT INT8 engines via explicit Q/DQ ONNX quantization.")
    parser.add_argument("--manifest", type=Path, default=Path("deploy/exports/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/exports"))
    parser.add_argument("--calibration-images", type=Path, default=Path("datasets/yolo_640/train/images"))
    parser.add_argument("--models", default="small,medium")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-per-channel", action="store_true")
    parser.add_argument("--trtexec-int8-flag", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = read_manifest(args.manifest)
    rows = selected_onnx_rows(manifest_rows, parse_csv(args.models))
    if not rows:
        raise RuntimeError("No RF-DETR ONNX rows found in manifest. Export/rebuild manifest first.")

    calibration_images = list_images(args.calibration_images, args.max_images, args.seed)
    for row in rows:
        run_name = row["run_name"]
        onnx_path = Path(row["artifact"])
        input_resolution = int(row.get("input_resolution") or 0)
        input_name, input_shape = onnx_input_info(onnx_path, input_resolution, args.batch)

        run_dir = args.output_dir / "rfdetr" / run_name / "int8_qdq"
        qdq_path = run_dir / "inference_model_int8_qdq.onnx"
        engine_path = run_dir / f"{run_name}_int8.engine"

        if not qdq_path.exists() or args.force:
            print(f"Quantizing Q/DQ ONNX: {onnx_path}")
            print(f"Input: {input_name} shape={input_shape}")
            print(f"Calibration images: {len(calibration_images)}")
            quantize_qdq(
                onnx_path=onnx_path,
                qdq_path=qdq_path,
                calibration_images=calibration_images,
                input_name=input_name,
                input_shape=input_shape,
                per_channel=not args.no_per_channel,
            )
            print(f"Q/DQ ONNX: {qdq_path}")

        if not engine_path.exists() or args.force:
            run_trtexec(qdq_path, engine_path, args.workspace, args.trtexec_int8_flag)

        if not manifest_has_artifact(args.manifest, engine_path):
            append_manifest(
                args.manifest,
                {
                    "family": "rfdetr",
                    "runtime": "tensorrt_engine",
                    "precision": "int8",
                    "quantization": "qdq",
                    "model": row["model"],
                    "run_name": run_name,
                    "artifact": str(engine_path.resolve()),
                    "source_onnx": str(onnx_path.resolve()),
                    "qdq_onnx": str(qdq_path.resolve()),
                    "calibration_images": str(args.calibration_images),
                    "calibration_max_images": args.max_images,
                    "input_resolution": row.get("input_resolution"),
                    "dataset_resolution": row.get("dataset_resolution"),
                },
            )
        print(f"RF-DETR Q/DQ INT8 engine: {engine_path}")

    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
