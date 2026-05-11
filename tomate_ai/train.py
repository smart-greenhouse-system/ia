
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

# ============================================================
# ENTRENAMIENTO DEL DETECTOR DE TOMATE CHERRY
# ============================================================
# Este script entrena un modelo YOLOv8 usando el dataset
# Cherry Tomato Plants descargado desde Roboflow.
#
# Además:
# - Guarda automáticamente todos los pesos.
# - Conserva el mejor modelo.
# - Conserva el último modelo.
# - Permite reanudar el entrenamiento más adelante.
# ============================================================

# Cargar modelo base preentrenado
# yolov8n.pt es ligero, rápido y excelente para comenzar.

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
# ============================================================
# ARCHIVOS GENERADOS
# ============================================================
#
# runs/detect/cherry_detector/
#
# ├── weights/
# │   ├── best.pt      -> Mejor modelo
# │   ├── last.pt      -> Última época
# │
# ├── results.csv      -> Métricas por época
# ├── results.png      -> Gráficas de entrenamiento
# ├── confusion_matrix.png
# └── args.yaml
#
# ============================================================

print("\nEntrenamiento finalizado correctamente.")
print("Mejor modelo: runs/detect/cherry_detector/weights/best.pt")
print("Último modelo: runs/detect/cherry_detector/weights/last.pt")