import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# ==========================================
# CARGAR VARIABLES .ENV
# ==========================================
load_dotenv(Path(__file__).parent / ".env")

# ==========================================
# API KEY
# ==========================================
rf = Roboflow(
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

# ==========================================
# PROYECTO
# ==========================================
project = rf.workspace(
    "bin-mibxd"
).project(
    "coin-segmentation-8ah7d"
)

# ==========================================
# VERSION DATASET
# ==========================================
version = project.version(1)

# ==========================================
# RUTA DESCARGA
# ==========================================
dataset_path = str(
    Path(__file__).parent / "dataset"
)

# ==========================================
# DESCARGAR DATASET
# ==========================================
dataset = version.download(
    "yolov8",
    location=dataset_path
)

print(
    "Dataset descargado en:",
    dataset.location
)