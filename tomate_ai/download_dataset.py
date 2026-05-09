# MODELO DE CRECIMIENTO DE TOMATE CHERRY
# from roboflow import Roboflow

# rf = Roboflow(api_key="API_ROBOTFLOW")

# project = rf.workspace("growth-stages").project(
#     "cherry-tomato-growth-stages-msmd7"
# )

# version = project.version(1)

# dataset = version.download("yolov8")

# print("Dataset descargado en:", dataset.location)

# MODELO DE DETECCIÓN DE PLANTAS DE TOMATE CHERRY
# from roboflow import Roboflow

# rf = Roboflow(api_key="ROBOTFLOW_API")

# project = rf.workspace("plantdetection-fcner").project(
#     "cherry-tomato-plants"
# )

# version = project.version(1)

# dataset = version.download("yolov8")

# print("Dataset descargado en:", dataset.location)


# MODELO DE SEGMENTACIÓN DE PLANTAS DE TOMATE CHERRY

from roboflow import Roboflow


rf = Roboflow(api_key="ROBOTFLOW_API")


rf = Roboflow(api_key="API_ROBOTFLOW")


rf = Roboflow(api_key="API_ROBOTFLOW")


project = rf.workspace("personal-6qa5a").project(
    "cherry-tomato-tpgsw"
)

version = project.version(4)

dataset = version.download("yolov8")


print("Dataset descargado en:", dataset.location)
