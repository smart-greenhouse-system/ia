import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
import torch

# =========================================================
# CONFIGURACIÓN BASE
# =========================================================

BASE_DIR = Path(__file__).parent

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =========================================================
# MODELO DE SEGMENTACIÓN DE TOMATES CHERRY
# =========================================================
"""
Entrenamiento para segmentación de tomates cherry.
"""

seg_model = YOLO("yolov8s-seg.pt")

seg_model.train(

    data=str(
        BASE_DIR
        / "dataset"
        / "data.yaml"
    ),

    task="segment",

    # Entrenamiento
    epochs=100,
    imgsz=640,
    batch=8,

    # GPU / CPU
    device=DEVICE,

    # Guardado
    save=True,
    save_period=5,

    # Proyecto
    project=str(BASE_DIR / "runs" / "segment"),
    name="tomato_growth",

    # Mejoras entrenamiento
    patience=20,
    pretrained=True,
    cache=True,

    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    # Optimizaciones
    amp=True,
    workers=8,

    # Visualización
    plots=True,
    verbose=True,

    exist_ok=True
)

# =========================================================
# MODELO DE CLASIFICACIÓN / DETECCIÓN
# =========================================================
"""
Entrenamiento para detectar estado del tomate cherry.
"""

detect_model = YOLO("yolov8n.pt")

detect_model.train(

    data=str(
        BASE_DIR
        / "cherry-tomato-classification-2"
        / "data.yaml"
    ),

    task="detect",

    # Entrenamiento
    epochs=50,
    imgsz=640,
    batch=16,

    # GPU / CPU
    device=DEVICE,

    # Guardado
    save=True,
    save_period=5,

    # Proyecto
    project=str(BASE_DIR / "runs" / "detect"),
    name="tomato_state_detector",

    # Configuración
    pretrained=True,
    patience=10,

    optimizer="AdamW",
    lr0=0.001,

    amp=True,
    workers=4,

    plots=True,
    verbose=True,

    exist_ok=True
)

# =========================================================
# OUTPUTS
# =========================================================

print("\nModelo de segmentación guardado en:")
print(
    BASE_DIR
    / "runs/segment/tomato_growth/weights/best.pt"
)

print("\nModelo de detección guardado en:")
print(
    BASE_DIR
    / "runs/detect/tomato_state_detector/weights/best.pt"
)