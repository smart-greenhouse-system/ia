import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(Path(__file__).parent / ".env")

# Workspace: omars-workspace-mjqor
# Proyecto:  lechuga-segmentacion
# Version:   2
# Clases:    etapas de crecimiento + tarjeta de referencia (clase 3)
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("omars-workspace-mjqor").project("lechuga-segmentacion")

version = project.version(2)

dataset_path = str(Path(__file__).parent / "dataset")
dataset = version.download("yolov8", location=dataset_path)

print("Dataset descargado en:", dataset.location)
