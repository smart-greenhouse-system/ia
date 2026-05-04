import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import MonedaDetector

BASE_DIR = Path(__file__).parent

detector = MonedaDetector(weights="yolov8n-seg.pt")

detector.train_model(
    data_yaml=str(BASE_DIR / "dataset" / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16
)
