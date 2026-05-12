# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent))

# from model import MonedaDetector

# BASE_DIR = Path(__file__).parent

# detector = MonedaDetector(weights="yolov8n-seg.pt")

# detector.train_model(

#     data_yaml=str(BASE_DIR / "dataset" / "data.yaml"),

#     data_yaml="coin-1/data.yaml",

#     epochs=50,
#     imgsz=640,
#     batch=16
# )


from ultralytics import YOLO

# ==========================================
# MODELO BASE SEGMENTACIÓN
# ==========================================
model = YOLO("yolov8n-seg.pt")

# ==========================================
# ENTRENAMIENTO
# ==========================================
model.train(

    # Dataset
    data="coin-1/data.yaml",

    # Entrenamiento
    epochs=100,
    imgsz=640,
    batch=8,

    # GPU
    device=0,

    # Optimización
    patience=20,
    pretrained=True,

    # Guardado
    save=True,
    save_period=5,

    # Proyecto
    project="runs/segment",
    name="coin_segmentation",

    # Segmentación
    task="segment"
)
