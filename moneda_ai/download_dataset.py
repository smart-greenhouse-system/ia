<<<<<<< HEAD
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(Path(__file__).parent / ".env")

from roboflow import Roboflow

rf = Roboflow(api_key="TQJwc8CVCEjpmYXuBbAt")


project = rf.workspace("bin-mibxd").project(
    "coin-segmentation-8ah7d"
)

version = project.version(1)

dataset = version.download("yolov8")


dataset_path = str(Path(__file__).parent / "dataset")
dataset = version.download("yolov8", location=dataset_path)


print("Dataset descargado en:", dataset.location)

