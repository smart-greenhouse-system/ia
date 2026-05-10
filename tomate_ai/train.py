import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

model = YOLO("yolov8s-seg.pt")

model.train(
    data=str(BASE_DIR / "dataset" / "data.yaml"),
    task="segment",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    save=True,
    save_period=5,
    project=str(BASE_DIR / "runs" / "segment"),
    name="tomato_growth",
    patience=20,
    pretrained=True,
    cache=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    amp=True,
    workers=8,
    plots=True,
    verbose=True
)
