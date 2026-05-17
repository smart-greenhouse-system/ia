# # import sys
# # from pathlib import Path

# # sys.path.insert(0, str(Path(__file__).parent))

# # from model import MonedaDetector

# # BASE_DIR = Path(__file__).parent

# # detector = MonedaDetector(weights="yolov8n-seg.pt")

# # detector.train_model(

# #     data_yaml=str(BASE_DIR / "dataset" / "data.yaml"),

# #     data_yaml="coin-1/data.yaml",

# #     epochs=50,
# #     imgsz=640,
# #     batch=16
# # )


# from ultralytics import YOLO

# # ==========================================
# # MODELO BASE SEGMENTACIÓN
# # ==========================================
# model = YOLO("yolov8n-seg.pt")

# # ==========================================
# # ENTRENAMIENTO
# # ==========================================
# model.train(

#     # Dataset
#     data="coin-1/data.yaml",

#     # Entrenamiento
#     epochs=100,
#     imgsz=640,
#     batch=8,

#     # GPU
#     device=0,

#     # Optimización
#     patience=20,
#     pretrained=True,

#     # Guardado
#     save=True,
#     save_period=5,

#     # Proyecto
#     project="runs/segment",
#     name="coin_segmentation",

#     # Segmentación
#     task="segment"
# )

from ultralytics import YOLO
from pathlib import Path

# ==========================================
# RUTAS
# ==========================================
BASE_DIR = Path(__file__).parent

DATASET_PATH = (
    BASE_DIR /
    "Card-SEG.v10-phonecardsegv1.yolov8" /
    "data.yaml"
)

# ==========================================
# MODELO BASE
# ==========================================
model = YOLO("yolov8n-seg.pt")

# ==========================================
# ENTRENAMIENTO
# ==========================================
model.train(

    # Dataset
    data=str(DATASET_PATH),

    # ======================================
    # HIPERPARÁMETROS
    # ======================================
    epochs=150,
    imgsz=640,
    batch=8,

    # ======================================
    # HARDWARE
    # ======================================
    device=0,          # GPU NVIDIA
    workers=4,
    cache=True,

    # ======================================
    # OPTIMIZACIÓN
    # ======================================
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    patience=30,

    # ======================================
    # REGULARIZACIÓN
    # ======================================
    dropout=0.05,

    # ======================================
    # GUARDADO
    # ======================================
    save=True,
    save_period=5,

    # ======================================
    # PROYECTO
    # ======================================
    project="runs/segment",
    name="phone_card_segmentation",

    # ======================================
    # REPRODUCIBILIDAD
    # ======================================
    seed=42,
    deterministic=True,

    # ======================================
    # VALIDACIÓN
    # ======================================
    val=True,
    plots=True,

    # ======================================
    # SEGMENTACIÓN
    # ======================================
    task="segment"
)
