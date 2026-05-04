import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("bin-mibxd").project("coin-segmentation-8ah7d")

version = project.version(4)

dataset = version.download("yolov8", location="dataset")

print("Dataset descargado en:", dataset.location)
