import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(Path(__file__).parent / ".env")

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("personal-6qa5a").project("cherry-tomato-tpgsw")

version = project.version(4)

dataset_path = str(Path(__file__).parent / "dataset")
dataset = version.download("yolov8", location=dataset_path)

print("Dataset descargado en:", dataset.location)