from roboflow import Roboflow

rf = Roboflow(api_key="API_ROBOTFLOW")

project = rf.workspace("growth-stages").project(
    "cherry-tomato-growth-stages-msmd7"
)

version = project.version(1)

dataset = version.download("yolov8")

print("Dataset descargado en:", dataset.location)