# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")

# model.train(
#     data="dataset/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     save=True,
#     save_period=1
# } 


from ultralytics import YOLO

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
model = YOLO("yolov8n.pt")

# Iniciar entrenamiento
results = model.train(
    # Archivo de configuración del dataset
    data="Cherry-Tomato-Plants-1/data.yaml",

    # Número total de épocas
    epochs=100,

    # Tamaño de entrada de las imágenes
    imgsz=640,

    # Número de imágenes por lote
    batch=16,

    # Nombre del experimento
    name="cherry_detector",

    # Carpeta principal donde se guardará
    project="runs/detect",

    # Guarda checkpoints automáticamente
    save=True,

    # Guardar un checkpoint cada 10 épocas
    save_period=10,

    # Mantener gráficos y métricas
    plots=True,

    # Permite sobrescribir si existe
    exist_ok=True,

    # Usa pesos preentrenados
    pretrained=True,

    # Número de workers para carga de datos
    workers=8,

    # Modo determinístico para reproducibilidad
    deterministic=True,

    # Semilla aleatoria
    seed=42
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