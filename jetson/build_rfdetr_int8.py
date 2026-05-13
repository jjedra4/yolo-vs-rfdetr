import argparse
import ctypes
import json
import random
from pathlib import Path

import numpy as np
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


class CudaRuntime:
    cudaMemcpyHostToDevice = 1

    def __init__(self) -> None:
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                self.lib = None
        if self.lib is None:
            raise RuntimeError("Could not load libcudart.so")

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        status = self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if status != 0:
            raise RuntimeError(f"cudaMalloc failed with status {status}")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr:
            self.lib.cudaFree(ctypes.c_void_p(ptr))

    def memcpy_htod(self, dst_ptr: int, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        src_ptr = contiguous.ctypes.data_as(ctypes.c_void_p)
        status = self.lib.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            src_ptr,
            ctypes.c_size_t(contiguous.nbytes),
            self.cudaMemcpyHostToDevice,
        )
        if status != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed with status {status}")
        self.lib.cudaDeviceSynchronize()


def preprocess_image(path: Path, height: int, width: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


class ImageEntropyCalibrator:
    def __init__(
        self,
        trt_module,
        images: list[Path],
        input_shape: tuple[int, int, int, int],
        input_name: str,
        cache_path: Path,
    ) -> None:
        self.trt = trt_module
        self.images = images
        self.batch_size, self.channels, self.height, self.width = input_shape
        if self.channels != 3:
            raise RuntimeError(f"Expected 3-channel RF-DETR input, got shape {input_shape}")
        self.input_name = input_name
        self.cache_path = cache_path
        self.index = 0
        self.cuda = CudaRuntime()
        self.host_batch = np.empty(input_shape, dtype=np.float32)
        self.device_input = self.cuda.malloc(self.host_batch.nbytes)

    def close(self) -> None:
        self.cuda.free(self.device_input)
        self.device_input = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index + self.batch_size > len(self.images):
            return None

        batch_paths = self.images[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        for batch_index, image_path in enumerate(batch_paths):
            self.host_batch[batch_index] = preprocess_image(image_path, self.height, self.width)

        self.cuda.memcpy_htod(self.device_input, self.host_batch)
        return [self.device_input]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(bytes(cache))


def make_calibrator_class(trt_module):
    class TensorRTImageEntropyCalibrator(trt_module.IInt8EntropyCalibrator2):
        def __init__(self, *args, **kwargs):
            trt_module.IInt8EntropyCalibrator2.__init__(self)
            self.impl = ImageEntropyCalibrator(trt_module, *args, **kwargs)

        def close(self) -> None:
            self.impl.close()

        def get_batch_size(self) -> int:
            return self.impl.get_batch_size()

        def get_batch(self, names):
            return self.impl.get_batch(names)

        def read_calibration_cache(self):
            return self.impl.read_calibration_cache()

        def write_calibration_cache(self, cache) -> None:
            self.impl.write_calibration_cache(cache)

    return TensorRTImageEntropyCalibrator


def set_workspace(config, trt_module, workspace_mib: int) -> None:
    workspace_bytes = int(workspace_mib) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt_module.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def set_builder_optimization_level(config, level: int | None) -> None:
    if level is None:
        return
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(level)
    elif hasattr(config, "set_builder_optimization_level"):
        config.set_builder_optimization_level(int(level))
    else:
        print("Warning: this TensorRT Python API does not expose builder optimization level.")


def set_calibrate_before_fusion(config, trt_module) -> None:
    flag = getattr(getattr(trt_module, "QuantizationFlag", object), "CALIBRATE_BEFORE_FUSION", None)
    if flag is None or not hasattr(config, "set_quantization_flag"):
        print("Warning: CALIBRATE_BEFORE_FUSION is not available in this TensorRT Python API.")
        return
    config.set_quantization_flag(flag)


def concrete_input_shape(network, batch_size: int, fallback_resolution: int) -> tuple[str, tuple[int, int, int, int]]:
    if network.num_inputs != 1:
        raise RuntimeError(f"Expected one RF-DETR ONNX input, got {network.num_inputs}")

    tensor = network.get_input(0)
    shape = list(tensor.shape)
    if len(shape) != 4:
        raise RuntimeError(f"Expected NCHW RF-DETR ONNX input, got {tensor.name}: {shape}")

    if shape[0] < 0:
        shape[0] = batch_size
    elif shape[0] != batch_size:
        raise RuntimeError(
            f"ONNX has static batch {shape[0]}, but --batch={batch_size}. "
            "Re-export ONNX with dynamic_batch=True or use the exported static batch."
        )
    if shape[1] < 0:
        shape[1] = 3
    if shape[2] < 0:
        shape[2] = fallback_resolution
    if shape[3] < 0:
        shape[3] = fallback_resolution

    shape[0] = batch_size
    return tensor.name, tuple(int(dim) for dim in shape)


def add_profile_if_dynamic(builder, network, config, input_name: str, input_shape: tuple[int, int, int, int]) -> None:
    tensor = network.get_input(0)
    if not any(dim < 0 for dim in tensor.shape):
        return
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)


def build_int8_engine(
    onnx_path: Path,
    engine_path: Path,
    calibration_images: list[Path],
    cache_path: Path,
    batch_size: int,
    fallback_resolution: int,
    workspace_mib: int,
    strict_int8: bool,
    builder_optimization_level: int | None,
    calibrate_before_fusion: bool,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX {onnx_path}:\n{errors}")

    input_name, input_shape = concrete_input_shape(network, batch_size, fallback_resolution)
    config = builder.create_builder_config()
    set_workspace(config, trt, workspace_mib)
    set_builder_optimization_level(config, builder_optimization_level)
    config.set_flag(trt.BuilderFlag.INT8)
    if not strict_int8:
        config.set_flag(trt.BuilderFlag.FP16)
    if calibrate_before_fusion:
        set_calibrate_before_fusion(config, trt)

    add_profile_if_dynamic(builder, network, config, input_name, input_shape)
    calibrator_cls = make_calibrator_class(trt)
    calibrator = calibrator_cls(calibration_images, input_shape, input_name, cache_path)
    config.int8_calibrator = calibrator

    print(f"Building INT8 engine: {onnx_path}")
    print(f"Input: {input_name} shape={input_shape}")
    print(f"Calibration images: {len(calibration_images)}")
    print(f"Calibration cache: {cache_path}")

    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(
                "TensorRT returned an empty serialized engine. If TensorRT 10.3 logged "
                "'region should have been removed from Graph::regions', this is a known "
                "TensorRT 10.3 INT8 calibration bug; try --builder-optimization-level 0 "
                "and --calibrate-before-fusion, otherwise use TensorRT 10.4+."
            )
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(serialized_engine))
    finally:
        calibrator.close()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibrated RF-DETR INT8 TensorRT engines from exported ONNX files.")
    parser.add_argument("--manifest", type=Path, default=Path("deploy/exports/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/exports"))
    parser.add_argument("--calibration-images", type=Path, default=Path("datasets/yolo_640/train/images"))
    parser.add_argument("--models", default="small,medium", help="Comma-separated RF-DETR model keys from manifest.")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=int, default=4096, help="TensorRT workspace MiB.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-int8", action="store_true", help="Do not allow FP16 fallback layers.")
    parser.add_argument("--builder-optimization-level", type=int, default=None)
    parser.add_argument("--calibrate-before-fusion", action="store_true")
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
        if input_resolution <= 0:
            raise RuntimeError(f"Missing input_resolution in manifest row: {run_name}")

        engine_dir = args.output_dir / "rfdetr" / run_name / "int8"
        engine_path = engine_dir / f"{run_name}_int8.engine"
        cache_path = engine_dir / f"{run_name}_int8.calib.cache"
        if engine_path.exists() and not args.force:
            print(f"Skipping existing engine: {engine_path}")
        else:
            build_int8_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                calibration_images=calibration_images,
                cache_path=cache_path,
                batch_size=args.batch,
                fallback_resolution=input_resolution,
                workspace_mib=args.workspace,
                strict_int8=args.strict_int8,
                builder_optimization_level=args.builder_optimization_level,
                calibrate_before_fusion=args.calibrate_before_fusion,
            )

        manifest_row = {
            "family": "rfdetr",
            "runtime": "tensorrt_engine",
            "precision": "int8",
            "model": row["model"],
            "run_name": run_name,
            "artifact": str(engine_path.resolve()),
            "calibration_cache": str(cache_path.resolve()),
            "calibration_images": str(args.calibration_images),
            "calibration_max_images": args.max_images,
            "input_resolution": row.get("input_resolution"),
            "dataset_resolution": row.get("dataset_resolution"),
        }
        if not any(
            existing.get("artifact") == manifest_row["artifact"]
            and existing.get("precision") == "int8"
            and existing.get("runtime") == "tensorrt_engine"
            for existing in read_manifest(args.manifest)
        ):
            append_manifest(args.manifest, manifest_row)
        print(f"RF-DETR INT8 engine: {engine_path}")

    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
