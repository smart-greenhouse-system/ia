import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import LechugaDetector

BASE_DIR = Path(__file__).parent

# Modelo base de segmentación YOLOv8 nano
# Dataset: lechuga-segmentacion v2 (Roboflow)
# Clases: etapas de lechuga + tarjeta de referencia (clase 3)
detector = LechugaDetector(weights="yolov8n-seg.pt")

detector.train_model(
    data_yaml=str(BASE_DIR / "dataset" / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    plots=True
)
