import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(Path(__file__).parent / ".env")

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("robotics-v9fnj").project("lettuce-growth-stage-qld0b")

version = project.version(1)

dataset_path = str(Path(__file__).parent / "dataset")
dataset = version.download("yolov8", location=dataset_path)

print("Dataset descargado en:", dataset.location)
