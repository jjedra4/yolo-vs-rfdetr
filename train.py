import os
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import clear_output, display

import comet_ml
from comet_ml import Experiment
import torch
from roboflow import Roboflow
from ultralytics import YOLO
from rfdetr import RFDETRSmall

# =========================
# API KEYS
# =========================
ROBOFLOW_API_KEY = "SxDKSK7rAUHKg69YSvkx"
COMET_API_KEY = "BCLjdlqC1KpDDoCwKQJDFHCjS"

# =========================
# DOWNLOAD DATASETS
# =========================
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("objectdetection-z2zn1").project("garbage-classification-3-bjcgp")

version_1280 = project.version(5)

dataset_YOLOv8_1280 = version_1280.download(
    "yolov8",
    location="./datasets/yolo_1280",
    overwrite=True
)

dataset_COCO_1280 = version_1280.download(
    "coco",
    location="./datasets/coco_1280",
    overwrite=True
)

version_640 = project.version(4)

dataset_YOLOv8_640 = version_640.download(
    "yolov8",
    location="./datasets/yolo_640",
    overwrite=True
)

dataset_COCO_640 = version_640.download(
    "coco",
    location="./datasets/coco_640",
    overwrite=True
)

version_416 = project.version(3)

dataset_YOLOv8_416 = version_416.download(
    "yolov8",
    location="./datasets/yolo_416",
    overwrite=True
)

dataset_COCO_416 = version_416.download(
    "coco",
    location="./datasets/coco_416",
    overwrite=True
)

# =========================
# CONFIG
# =========================
comet_ml.init(
    project_name="garbage-classification-comparison",
    api_key=COMET_API_KEY
)

datasets_yolo = {
    416: dataset_YOLOv8_416.location + "/data.yaml",
    640: dataset_YOLOv8_640.location + "/data.yaml",
    1280: dataset_YOLOv8_1280.location + "/data.yaml",
}

datasets_rfdetr = {
    416: dataset_COCO_416.location,
    640: dataset_COCO_640.location,
    1280: dataset_COCO_1280.location,
}

EPOCHS = 1
SEED = 42

# =========================
# TRAIN LOOP for YOLO models
# =========================
model_types = ["yolov8m.pt", "yolo11m.pt"]
for model_name in model_types:
    for res, data_path in datasets_yolo.items():
        print("=" * 60)
        print(f"\n--- START: {model_name} @ Rozdzielczość {res} ---")
        print("=" * 60)

        model = YOLO(model_name)

        experiment_name = f"{model_name.split('.')[0]}_res{res}"

        model.train(
            data=data_path,
            epochs=EPOCHS,
            imgsz=res,
            batch=0.8,
            name=experiment_name,
            project="Garbage_AI_Comparison",
            exist_ok=True,
            plots=True,
            val=True,
            seed=SEED,
        )

        print(f"--- KONIEC: {experiment_name} ---\n")


# =========================
# TRAIN LOOP for RF-DETR
# =========================

for res, dataset_path in datasets_rfdetr.items():

    experiment_name = f"rfdetr_small_{res}"

    print("=" * 60)
    print(f"Training: {experiment_name}")
    print("=" * 60)

    experiment = Experiment(project_name="garbage-ai-comparison")

    experiment.set_name(experiment_name)

    experiment.log_parameters(
        {
            "model": "RFDETRSmall",
            "resolution": res,
            "epochs": EPOCHS,
        }
    )

    model = RFDETRSmall()

    output_dir = f"runs/rfdetr/{experiment_name}"

    model.train(
        dataset_dir=dataset_path,
        epochs=EPOCHS,
        batch_size="auto",
        image_size=res,
        output_dir=output_dir,
        seed=SEED,
    )

    results_csv = Path(output_dir) / "results.csv"

    if results_csv.exists():

        df = pd.read_csv(results_csv)

        # compute F1
        if "precision" in df.columns and "recall" in df.columns:
            precision = df["precision"]
            recall = df["recall"]

            df["F1"] = 2 * precision * recall / (precision + recall + 1e-8)

        # log metrics to Comet per epoch
        for _, row in df.iterrows():
            epoch = int(row["epoch"])
            for col in df.columns:

                if col == "epoch":
                    continue

                value = row[col]

                if pd.notna(value):
                    experiment.log_metric(col, value, step=epoch)

    experiment.end()
    print(f"Finished: {experiment_name}")
