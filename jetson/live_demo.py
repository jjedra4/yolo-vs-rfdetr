#!/usr/bin/env python3
import argparse
import json
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_YOLO_RUN = "yolov8m_data640_img640"
DEFAULT_RFDETR_RUN = "rfdetr_small_data640_img512"
DEFAULT_BOUNDARY = "yolo_rfdetr_demo"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    score: float
    class_id: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def video_devices() -> list[str]:
    return [str(path) for path in sorted(Path("/dev").glob("video*"))]


def nvargus_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width=(int){width},height=(int){height},"
        f"format=(string)NV12,framerate=(fraction){fps}/1 ! "
        "nvvidconv ! "
        f"video/x-raw,width=(int){width},height=(int){height},format=(string)BGRx ! "
        "videoconvert ! video/x-raw,format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_existing_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if path.is_absolute():
        candidates.append(root / path.name)
        parts = path.parts
        if "deploy" in parts:
            deploy_index = parts.index("deploy")
            candidates.append(root / Path(*parts[deploy_index:]))
            candidates.append(root / "deploy" / Path(*parts[deploy_index:]))
        if "trained_weights" in parts:
            weights_index = parts.index("trained_weights")
            candidates.append(root / Path(*parts[weights_index:]))
    else:
        candidates.append(root / path)
        candidates.append(root / "deploy" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def select_manifest_row(
    manifest_path: Path,
    family: str,
    run_name: str,
    precision: str = "fp16",
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    for row in rows:
        if (
            row.get("family") == family
            and row.get("run_name") == run_name
            and row.get("precision") == precision
        ):
            return row
    available = ", ".join(
        f"{row.get('family')}:{row.get('run_name')}:{row.get('precision')}"
        for row in rows
    )
    raise RuntimeError(
        f"Model not found in {manifest_path}: {family}:{run_name}:{precision}. "
        f"Available: {available}"
    )


def load_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return list(names)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (42, 157, 143),
        (231, 111, 81),
        (38, 70, 83),
        (233, 196, 106),
        (58, 134, 255),
        (131, 56, 236),
        (255, 0, 110),
        (76, 201, 240),
    ]
    return palette[class_id % len(palette)]


def display_model_name(run_name: str) -> str:
    labels = {
        "yolov8s_data640_img640": "YOLOv8-S",
        "yolov8m_data640_img640": "YOLOv8-M",
        "yolo11s_data640_img640": "YOLO11-S",
        "yolo11m_data640_img640": "YOLO11-M",
        "rfdetr_small_data640_img512": "RF-DETR-S",
        "rfdetr_medium_data640_img576": "RF-DETR-M",
    }
    return labels.get(run_name, run_name)


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    class_names: list[str],
    title: str,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    header_height = 34

    header = f"{title} | {fps:.1f} FPS | {len(detections)} det."
    cv2.rectangle(annotated, (0, 0), (width, header_height), (18, 24, 30), -1)
    cv2.putText(
        annotated,
        header,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        color = color_for_class(det.class_id)
        label_name = class_names[det.class_id] if det.class_id < len(class_names) else str(det.class_id)
        label = f"{label_name} {det.score:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        text_w, text_h = text_size
        label_x = max(0, min(width - text_w - 7, x1))
        y_text = max(header_height + text_h + baseline + 5, y1 + text_h + baseline + 3)
        y_text = min(height - baseline - 2, y_text)
        cv2.rectangle(
            annotated,
            (label_x, y_text - text_h - baseline - 4),
            (min(width - 1, label_x + text_w + 6), y_text + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 3, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def filter_large_detections(
    detections: list[Detection],
    width: int,
    height: int,
    max_area_ratio: float,
    max_width_ratio: float,
    max_height_ratio: float,
) -> list[Detection]:
    if (
        (max_area_ratio <= 0.0 or max_area_ratio >= 1.0)
        and (max_width_ratio <= 0.0 or max_width_ratio >= 1.0)
        and (max_height_ratio <= 0.0 or max_height_ratio >= 1.0)
    ):
        return detections
    frame_area = float(width * height)
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        clipped_x1 = max(0, min(width - 1, x1))
        clipped_x2 = max(0, min(width - 1, x2))
        clipped_y1 = max(0, min(height - 1, y1))
        clipped_y2 = max(0, min(height - 1, y2))
        box_width = max(0, clipped_x2 - clipped_x1)
        box_height = max(0, clipped_y2 - clipped_y1)
        box_area = box_width * box_height
        too_large = (
            (0.0 < max_area_ratio < 1.0 and box_area / frame_area > max_area_ratio)
            or (0.0 < max_width_ratio < 1.0 and box_width / width > max_width_ratio)
            or (0.0 < max_height_ratio < 1.0 and box_height / height > max_height_ratio)
        )
        if not too_large:
            filtered.append(det)
    return filtered


def encode_status_frame(message: str, width: int = 1280, height: int = 360) -> bytes:
    frame = np.full((height, width, 3), (18, 24, 30), dtype=np.uint8)
    cv2.putText(
        frame,
        "YOLO vs RF-DETR Jetson Demo",
        (32, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    lines = split_status_text(message, max_chars=88)
    y = 136
    for line in lines[:5]:
        cv2.putText(
            frame,
            line,
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (233, 196, 106),
            2,
            cv2.LINE_AA,
        )
        y += 38
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else b""


def split_status_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        if current and current_len + len(word) + 1 > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
    if current:
        lines.append(" ".join(current))
    return lines or [text]


class YoloEngineDetector:
    def __init__(self, artifact: Path, input_resolution: int, conf: float, iou: float, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(artifact))
        self.input_resolution = input_resolution
        self.conf = conf
        self.iou = iou
        self.device = device

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame_bgr,
            imgsz=self.input_resolution,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, classes):
            detections.append(
                Detection(
                    xyxy=tuple(int(round(value)) for value in box[:4]),
                    score=float(score),
                    class_id=int(class_id),
                )
            )
        return detections


class RfDetrTensorRtDetector:
    def __init__(
        self,
        artifact: Path,
        input_resolution: int,
        conf: float,
        class_count: int,
        nms_iou: float,
        max_detections: int,
    ):
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        self.input_resolution = input_resolution
        self.conf = conf
        self.class_count = class_count
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.runner = TrtRunner(EngineFromBytes(bytes_from_path(artifact)))
        self.runner.activate()
        input_metadata = self.runner.get_input_metadata()
        self.input_name = next(iter(input_metadata.keys()))

    def close(self) -> None:
        self.runner.deactivate()

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        input_tensor = self.preprocess(frame_bgr)
        outputs = self.runner.infer({self.input_name: input_tensor})
        return self.postprocess(outputs, frame_bgr.shape[1], frame_bgr.shape[0])

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self.input_resolution, self.input_resolution), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1))
        return np.ascontiguousarray(chw[None, ...], dtype=np.float32)

    def postprocess(self, outputs: dict[str, np.ndarray], width: int, height: int) -> list[Detection]:
        arrays = {name: np.asarray(value) for name, value in outputs.items()}
        boxes = find_boxes_array(arrays)
        logits = find_logits_array(arrays, boxes)

        if boxes is None or logits is None:
            raise RuntimeError(f"Could not identify RF-DETR output tensors. Outputs: {describe_outputs(arrays)}")

        boxes = np.squeeze(boxes)
        logits = np.squeeze(logits)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise RuntimeError(f"Unexpected RF-DETR box output shape: {boxes.shape}")
        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected RF-DETR logits output shape: {logits.shape}")

        scores_by_class = sigmoid(logits) if logits.min() < 0.0 or logits.max() > 1.0 else logits
        if scores_by_class.shape[-1] == self.class_count + 1:
            scores_by_class = scores_by_class[:, : self.class_count]
        elif scores_by_class.shape[-1] > self.class_count:
            scores_by_class = scores_by_class[:, : self.class_count]

        class_ids = np.argmax(scores_by_class, axis=1)
        scores = scores_by_class[np.arange(scores_by_class.shape[0]), class_ids]
        keep = scores >= self.conf
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if len(boxes) == 0:
            return []

        xyxy = cxcywh_to_xyxy(boxes)
        xyxy[:, [0, 2]] *= width
        xyxy[:, [1, 3]] *= height
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)

        keep_indices = nms(xyxy, scores, self.nms_iou, self.max_detections)
        detections = []
        for idx in keep_indices:
            detections.append(
                Detection(
                    xyxy=tuple(int(round(value)) for value in xyxy[idx]),
                    score=float(scores[idx]),
                    class_id=int(class_ids[idx]),
                )
            )
        return detections


def bytes_from_path(path: Path):
    def loader() -> bytes:
        return path.read_bytes()

    return loader


def find_boxes_array(outputs: dict[str, np.ndarray]) -> np.ndarray | None:
    named = [value for name, value in outputs.items() if "box" in name.lower() and value.shape[-1:] == (4,)]
    if named:
        return named[0]
    for value in outputs.values():
        if value.ndim >= 2 and value.shape[-1] == 4:
            return value
    return None


def find_logits_array(outputs: dict[str, np.ndarray], boxes: np.ndarray | None) -> np.ndarray | None:
    named = [
        value
        for name, value in outputs.items()
        if any(token in name.lower() for token in ("logit", "score", "class")) and value is not boxes
    ]
    if named:
        return named[0]
    for value in outputs.values():
        if value is boxes:
            continue
        if value.ndim >= 2 and value.shape[-1] != 4:
            return value
    return None


def describe_outputs(outputs: dict[str, np.ndarray]) -> str:
    return ", ".join(f"{name}={tuple(value.shape)}:{value.dtype}" for name, value in outputs.items())


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[idx] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


class DemoState:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg: bytes | None = encode_status_frame("Starting models and opening camera...")
        self.stats = {"frames": 0, "error": ""}
        self.stopped = False

    def update(self, jpeg: bytes, stats: dict[str, Any]) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.stats = stats
            self.condition.notify_all()

    def set_error(self, error: str) -> None:
        with self.condition:
            self.jpeg = encode_status_frame(error)
            self.stats = {**self.stats, "error": error}
            self.condition.notify_all()


class DemoWorker(threading.Thread):
    def __init__(self, args: argparse.Namespace, state: DemoState):
        super().__init__(daemon=True)
        self.args = args
        self.state = state
        self.rfdetr_detector: RfDetrTensorRtDetector | None = None

    def run(self) -> None:
        root = repo_root()
        class_names = load_class_names(self.args.data_yaml)
        yolo_row = select_manifest_row(self.args.manifest, "yolo", self.args.yolo_run, "fp16")
        rfdetr_row = select_manifest_row(self.args.manifest, "rfdetr", self.args.rfdetr_run, "fp16")
        yolo_title = f"{display_model_name(yolo_row['run_name'])} FP16"
        rfdetr_title = f"{display_model_name(rfdetr_row['run_name'])} FP16"
        yolo_artifact = resolve_existing_path(yolo_row["artifact"], root)
        rfdetr_artifact = resolve_existing_path(rfdetr_row["artifact"], root)

        yolo_detector = YoloEngineDetector(
            yolo_artifact,
            int(yolo_row["input_resolution"]),
            self.args.yolo_conf,
            self.args.iou,
            self.args.device,
        )
        self.rfdetr_detector = RfDetrTensorRtDetector(
            rfdetr_artifact,
            int(rfdetr_row["input_resolution"]),
            self.args.rfdetr_conf,
            len(class_names),
            self.args.iou,
            self.args.max_detections,
        )

        capture = None
        try:
            capture = open_capture(
                self.args.camera,
                self.args.width,
                self.args.height,
                self.args.fps,
                self.args.fourcc,
                self.args.sensor_id,
            )
            while not self.state.stopped:
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.state.set_error("Camera frame read failed")
                    time.sleep(0.2)
                    continue

                if self.args.flip is not None:
                    frame = cv2.flip(frame, self.args.flip)

                if self.args.display_width and frame.shape[1] > self.args.display_width:
                    scale = self.args.display_width / frame.shape[1]
                    frame = cv2.resize(frame, (self.args.display_width, int(frame.shape[0] * scale)))

                yolo_start = time.perf_counter()
                yolo_detections = yolo_detector.predict(frame)
                yolo_detections = filter_large_detections(
                    yolo_detections,
                    frame.shape[1],
                    frame.shape[0],
                    self.args.max_box_area,
                    self.args.max_box_width,
                    self.args.max_box_height,
                )
                yolo_elapsed = time.perf_counter() - yolo_start

                rfdetr_start = time.perf_counter()
                rfdetr_detections = self.rfdetr_detector.predict(frame)
                rfdetr_detections = filter_large_detections(
                    rfdetr_detections,
                    frame.shape[1],
                    frame.shape[0],
                    self.args.max_box_area,
                    self.args.max_box_width,
                    self.args.max_box_height,
                )
                rfdetr_elapsed = time.perf_counter() - rfdetr_start

                left = draw_detections(frame, yolo_detections, class_names, yolo_title, 1.0 / yolo_elapsed)
                right = draw_detections(frame, rfdetr_detections, class_names, rfdetr_title, 1.0 / rfdetr_elapsed)
                combined = np.hstack([left, right])
                ok, encoded = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
                if ok:
                    self.state.update(
                        encoded.tobytes(),
                        {
                            "frames": self.state.stats.get("frames", 0) + 1,
                            "error": "",
                            "yolo_ms": yolo_elapsed * 1000.0,
                            "rfdetr_ms": rfdetr_elapsed * 1000.0,
                        },
                    )
        except Exception as exc:
            self.state.set_error(str(exc))
            traceback.print_exc()
            raise
        finally:
            if capture is not None:
                capture.release()
            if self.rfdetr_detector is not None:
                self.rfdetr_detector.close()


def open_capture(
    camera: str,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None,
    sensor_id: int,
) -> cv2.VideoCapture:
    if camera == "csi":
        pipeline = nvargus_pipeline(sensor_id, width, height, fps)
        print(f"Using nvargus camera pipeline: {pipeline}")
        return open_single_capture(pipeline, width, height, fps, fourcc)

    if camera == "auto":
        errors = []
        for candidate in video_devices():
            try:
                capture = open_single_capture(candidate, width, height, fps, fourcc)
                ok, frame = capture.read()
                if ok and frame is not None:
                    print(f"Using camera: {candidate}")
                    return capture
                capture.release()
                errors.append(f"{candidate}: opened but no frame")
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("No working V4L2 camera found. " + "; ".join(errors))

    return open_single_capture(camera, width, height, fps, fourcc)


def open_single_capture(camera: str, width: int, height: int, fps: int, fourcc: str | None) -> cv2.VideoCapture:
    source: int | str = int(camera) if camera.isdigit() else camera
    is_gstreamer = isinstance(source, str) and "!" in source
    if is_gstreamer:
        capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2 if isinstance(source, int) else cv2.CAP_ANY)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera source: {camera}")
    if is_gstreamer:
        return capture
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if not getattr(self.server, "quiet", False):
                super().log_message(fmt, *args)

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index"):
                self.send_html()
            elif self.path.startswith("/stream"):
                self.send_stream()
            elif self.path.startswith("/health"):
                self.send_json(state.stats)
            elif self.path.startswith("/favicon.ico"):
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def send_html(self) -> None:
            body = b"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO vs RF-DETR Jetson Demo</title>
  <style>
    html, body { margin: 0; background: #101418; color: #e8edf2; font-family: system-ui, sans-serif; }
    header { height: 44px; display: flex; align-items: center; padding: 0 16px; background: #182029; font-weight: 650; }
    main { height: calc(100vh - 44px); display: grid; place-items: center; }
    img { max-width: 100%; max-height: 100%; object-fit: contain; }
  </style>
</head>
<body>
  <header>YOLO vs RF-DETR Jetson Demo</header>
  <main><img src="/stream" alt="Jetson object detection stream"></main>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_stream(self) -> None:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={DEFAULT_BOUNDARY}")
            self.end_headers()
            while not state.stopped:
                with state.condition:
                    state.condition.wait(timeout=2.0)
                    jpeg = state.jpeg
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(f"--{DEFAULT_BOUNDARY}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

    return Handler


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Live YOLO vs RF-DETR Jetson browser demo.")
    parser.add_argument("--manifest", type=Path, default=root / "deploy" / "exports" / "manifest.jsonl")
    parser.add_argument("--data-yaml", type=Path, default=root / "datasets" / "yolo_640" / "data.yaml")
    parser.add_argument("--yolo-run", default=DEFAULT_YOLO_RUN)
    parser.add_argument("--rfdetr-run", default=DEFAULT_RFDETR_RUN)
    parser.add_argument(
        "--camera",
        default="csi",
        help="csi for nvarguscamerasrc, auto for V4L2 scan, OpenCV index, video path, RTSP URL, or GStreamer pipeline",
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI camera sensor-id for nvarguscamerasrc")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="", help="Optional V4L2 pixel format, for example MJPG or YUYV")
    parser.add_argument("--display-width", type=int, default=960, help="Downscale camera frame before inference/display")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--yolo-conf", type=float, default=None, help="YOLO confidence threshold; defaults to --conf")
    parser.add_argument("--rfdetr-conf", type=float, default=None, help="RF-DETR confidence threshold; defaults to --conf")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=0.75,
        help="Drop detections whose box area exceeds this fraction of the frame; use 1.0 to disable",
    )
    parser.add_argument(
        "--max-box-width",
        type=float,
        default=0.95,
        help="Drop detections whose box width exceeds this fraction of the frame; use 1.0 to disable",
    )
    parser.add_argument(
        "--max-box-height",
        type=float,
        default=0.95,
        help="Drop detections whose box height exceeds this fraction of the frame; use 1.0 to disable",
    )
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--flip", type=int, default=None, choices=[-1, 0, 1])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.yolo_conf is None:
        args.yolo_conf = args.conf
    if args.rfdetr_conf is None:
        args.rfdetr_conf = args.conf
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_yaml}")

    state = DemoState()
    worker = DemoWorker(args, state)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.quiet = args.quiet
    server.timeout = 0.5

    def stop(_signum=None, _frame=None) -> None:
        state.stopped = True
        with state.condition:
            state.condition.notify_all()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"Demo URL: http://{args.host}:{args.port}/")
    try:
        while not state.stopped:
            server.handle_request()
    finally:
        state.stopped = True
        with state.condition:
            state.condition.notify_all()
        worker.join(timeout=5)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
