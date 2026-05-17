# import os
# from pathlib import Path

# from dotenv import load_dotenv
# from roboflow import Roboflow

# # ==========================================
# # CARGAR VARIABLES .ENV
# # ==========================================
# load_dotenv(Path(__file__).parent / ".env")

# # ==========================================
# # API KEY
# # ==========================================
# rf = Roboflow(
#     api_key=os.environ["ROBOFLOW_API_KEY"]
# )

# # ==========================================
# # PROYECTO
# # ==========================================
# project = rf.workspace(
#     "bin-mibxd"
# ).project(
#     "coin-segmentation-8ah7d"
# )

# # ==========================================
# # VERSION DATASET
# # ==========================================
# version = project.version(1)

# # ==========================================
# # RUTA DESCARGA
# # ==========================================
# dataset_path = str(
#     Path(__file__).parent / "dataset"
# )

# # ==========================================
# # DESCARGAR DATASET
# # ==========================================
# dataset = version.download(
#     "yolov8",
#     location=dataset_path
# )

# print(
#     "Dataset descargado en:",
#     dataset.location
# )

# MODELO DE SEGMENTACIÓN DE MONEDAS

# from roboflow import Roboflow

# rf = Roboflow(
#     api_key="API_KEY"
# )

# project = rf.workspace(
#     "buu-ovf24"
# ).project(
#     "coin-1nyzl"
# )

# version = project.version(1)

# dataset = version.download(
#     "yolov8"
# )

# print(dataset.location)

# MODELO DE SEGMENTACION DE CARTAS POKEMON
from pathlib import Path
from roboflow import Roboflow

# Carpeta actual del proyecto
BASE_DIR = Path(__file__).parent

# Conexión con Roboflow
rf = Roboflow(
    api_key="TQJwc8CVCEjpmYXuBbAt"
)

# Proyecto
project = rf.workspace(
    "nandersen"
).project(
    "card-seg-j74w1"
)

# Versión del dataset
version = project.version(10)

# Descargar dataset YOLOv8
dataset = version.download(
    "yolov8",
    location=str(BASE_DIR)
)

print("Dataset descargado en:")
print(dataset.location)

