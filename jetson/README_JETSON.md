# Jetson Orin Nano Deployment and Benchmark

Target device:

```bash
ssh jetson@192.168.0.162
```

TensorRT `.engine` files must be built on the Jetson. Do not build engines on the desktop PC and copy them over.

## 1. Copy Project Data From PC

Run from the PC in the repo root:

```bash
rsync -av --progress \
  jetson \
  requirements-jetson.txt \
  trained_weights \
  jetson@192.168.0.162:~/yolo-vs-rfdetr/
ssh jetson@192.168.0.162 "mkdir -p ~/yolo-vs-rfdetr/runs/train_compare"
rsync -av --progress runs/train_compare/summary.jsonl jetson@192.168.0.162:~/yolo-vs-rfdetr/runs/train_compare/summary.jsonl
ssh jetson@192.168.0.162 "mkdir -p ~/yolo-vs-rfdetr/datasets/yolo_640"
rsync -av --progress datasets/yolo_640/ jetson@192.168.0.162:~/yolo-vs-rfdetr/datasets/yolo_640/
```

## 2. Prepare Jetson

Run on Jetson:

```bash
ssh jetson@192.168.0.162
cd ~/yolo-vs-rfdetr
```

Set a stable high-performance mode before benchmarking:

```bash
sudo nvpmodel -q
sudo nvpmodel -m 0
sudo jetson_clocks
```

Use the global JetPack PyTorch installation directly. Do not create a venv if PyTorch is already installed globally.

First remove any user-site PyTorch that could shadow the global JetPack build:

```bash
python3 -m pip uninstall --user -y torch torchvision torchaudio
```

Install only the missing project dependencies. This requirements file does not list PyTorch:

```bash
python3 -m pip install --upgrade --user pip
python3 -m pip install --user -r requirements-jetson.txt
```

Verify CUDA/TensorRT basics:

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

which trtexec
tegrastats --interval 1000
```

Stop `tegrastats` with `Ctrl+C`.

## 3. PyTorch Baseline Benchmark

This benchmarks YOLO `.pt` and RF-DETR `.pth` end-to-end:

```bash
python jetson/benchmark_models.py \
  --summary runs/train_compare/summary.jsonl \
  --images datasets/yolo_640/valid/images \
  --output-dir jetson_results/pytorch_baseline \
  --include-runtime ultralytics_pt,rfdetr_pytorch \
  --warmup 10 \
  --repeats 3
```

Quick smoke test:

```bash
python jetson/benchmark_models.py \
  --summary runs/train_compare/summary.jsonl \
  --images datasets/yolo_640/valid/images \
  --output-dir jetson_results/smoke_pytorch \
  --include-runtime ultralytics_pt,rfdetr_pytorch \
  --limit 20 \
  --warmup 3 \
  --repeats 1
```

## 4. Export YOLO TensorRT FP16 and INT8

Build engines on Jetson:

```bash
python jetson/export_models.py \
  --summary runs/train_compare/summary.jsonl \
  --include yolo \
  --yolo-precisions fp16,int8 \
  --data-yaml datasets/yolo_640/data.yaml \
  --output-dir deploy/exports \
  --device 0 \
  --batch 1 \
  --workspace 4096 \
  --force
```

Benchmark exported YOLO engines:

```bash
python jetson/benchmark_models.py \
  --manifest deploy/exports/manifest.jsonl \
  --images datasets/yolo_640/valid/images \
  --output-dir jetson_results/yolo_tensorrt \
  --include-runtime ultralytics_engine \
  --warmup 10 \
  --repeats 3
```

## 5. Export RF-DETR ONNX / TensorRT FP16

RF-DETR PyTorch inference is benchmarked in step 3. The export script can also create ONNX and FP16 TensorRT engines:

```bash
python jetson/export_models.py \
  --summary runs/train_compare/summary.jsonl \
  --include rfdetr \
  --rfdetr-precisions onnx,fp16 \
  --output-dir deploy/exports \
  --device 0 \
  --batch 1 \
  --workspace 4096 \
  --force
```

RF-DETR INT8 is intentionally disabled by default. It needs a proper calibration/QDQ path before the numbers are meaningful.

## 6. Results

Each benchmark directory contains:

```text
benchmark_summary.csv
raw_latency.csv
tegrastats/*.log
```

Key columns:

```text
fps_mean
latency_mean_ms
latency_median_ms
latency_p95_ms
latency_p99_ms
tegrastats_*_mean
tegrastats_*_max
```

Copy results back to PC:

```bash
rsync -av --progress jetson@192.168.0.162:~/yolo-vs-rfdetr/jetson_results ./jetson_results
rsync -av --progress jetson@192.168.0.162:~/yolo-vs-rfdetr/deploy ./deploy
```
