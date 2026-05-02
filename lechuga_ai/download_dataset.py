import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("robotics-v9fnj").project("lettuce-growth-stage-qld0b")

version = project.version(1)

dataset = version.download("yolov8", location="dataset")

print("Dataset descargado en:", dataset.location)
