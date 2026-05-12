# from ultralytics import YOLO
# ENTRENAR MODELO DE CRECIMIENTO DE TOMATE CHERRY
# model = YOLO("yolov8n.pt")

# model.train(
#     data="dataset/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     save=True,
#     save_period=1
# } 


# from ultralytics import YOLO

# # ============================================================
# # ENTRENAMIENTO DEL DETECTOR DE TOMATE CHERRY
# # ============================================================
# # Este script entrena un modelo YOLOv8 usando el dataset
# # Cherry Tomato Plants descargado desde Roboflow.
# #
# # Además:
# # - Guarda automáticamente todos los pesos.
# # - Conserva el mejor modelo.
# # - Conserva el último modelo.
# # - Permite reanudar el entrenamiento más adelante.
# # ============================================================

# # Cargar modelo base preentrenado
# # yolov8n.pt es ligero, rápido y excelente para comenzar.
# model = YOLO("yolov8n.pt")

# # Iniciar entrenamiento
# results = model.train(
#     # Archivo de configuración del dataset
#     data="Cherry-Tomato-Plants-1/data.yaml",

#     # Número total de épocas
#     epochs=100,

#     # Tamaño de entrada de las imágenes
#     imgsz=640,

#     # Número de imágenes por lote
#     batch=16,

#     # Nombre del experimento
#     name="cherry_detector",

#     # Carpeta principal donde se guardará
#     project="runs/detect",

#     # Guarda checkpoints automáticamente
#     save=True,

#     # Guardar un checkpoint cada 10 épocas
#     save_period=10,

#     # Mantener gráficos y métricas
#     plots=True,

#     # Permite sobrescribir si existe
#     exist_ok=True,

#     # Usa pesos preentrenados
#     pretrained=True,

#     # Número de workers para carga de datos
#     workers=8,

#     # Modo determinístico para reproducibilidad
#     deterministic=True,

#     # Semilla aleatoria
#     seed=42
# )

# # ============================================================
# # ARCHIVOS GENERADOS
# # ============================================================
# #
# # runs/detect/cherry_detector/
# #
# # ├── weights/
# # │   ├── best.pt      -> Mejor modelo
# # │   ├── last.pt      -> Última época
# # │
# # ├── results.csv      -> Métricas por época
# # ├── results.png      -> Gráficas de entrenamiento
# # ├── confusion_matrix.png
# # └── args.yaml
# #
# # ============================================================

# print("\nEntrenamiento finalizado correctamente.")
# print("Mejor modelo: runs/detect/cherry_detector/weights/best.pt")
# print("Último modelo: runs/detect/cherry_detector/weights/last.pt")

# MODELO DE SEGMENTACIÓN DE PLANTAS DE TOMATE CHERRY

# """ from ultralytics import YOLO

# # Modelo base
# model = YOLO("yolov8s-seg.pt")

# # Entrenamiento
# model.train(
#     data="cherry-tomato-4/data.yaml",
#     task="segment",

#     # Entrenamiento
#     epochs=100,
#     imgsz=640,
#     batch=8,

#     # GPU
#     device=0,

#     # Guardado
#     save=True,
#     save_period=5,

#     # Proyecto
#     project="runs/segment",
#     name="tomato_growth",

#     # Mejoras entrenamiento
#     patience=20,
#     pretrained=True,
#     cache=True,

#     # Data augmentation
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,

#     degrees=10,
#     translate=0.1,
#     scale=0.5,
#     fliplr=0.5,

#     # Optimizaciones
#     amp=True,
#     workers=8,

#     # Visualización
#     plots=True,
#     verbose=True
# ) """


# MODELO DE CLASIFICACION DE TOMATE CHERRY

from ultralytics import YOLO
from pathlib import Path
import torch

BASE_DIR = Path(
    "/home/pablo/Documents/Python/crecimiento/tomate_ai"
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(

    data=str(
        BASE_DIR
        / "cherry-tomato-classification-2"
        / "data.yaml"
    ),

    task="detect",

    epochs=50,
    imgsz=640,
    batch=16,

    device=DEVICE,

    project=str(BASE_DIR / "runs/detect"),
    name="tomato_state_detector",

    save=True,
    save_period=5,

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

print("\nModelo guardado en:")
print(
    BASE_DIR
    / "runs/detect/tomato_state_detector/weights/best.pt"
)