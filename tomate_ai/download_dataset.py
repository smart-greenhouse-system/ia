
# 1. Clasificar el tomate cherry https://universe.roboflow.com/cherrytomato-n6w1z/cherry-tomato-classification-weatf/dataset/2
# 2. Detectar plantas cherry-tomato-plants https://universe.roboflow.com/plantdetection-fcner/cherry-tomato-plants
# 3. Detectar etapa de crecimiento https://universe.roboflow.com/growth-stages/cherry-tomato-growth-stages-msmd7
# 4. Segmentar plantas cherry-tomato-tpgsw https://universe.roboflow.com/personal-6qa5a/cherry-tomato-tpgsw/dataset/4
# 5. Calcular tamaño de la planta usando la segmentacion y la escala real (diámetro de una moneda) (falta implementar)

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

# from roboflow import Roboflow


# rf = Roboflow(api_key="ROBOTFLOW_API")


# rf = Roboflow(api_key="API_ROBOTFLOW")


# rf = Roboflow(api_key="API_ROBOTFLOW")


# project = rf.workspace("personal-6qa5a").project(
#     "cherry-tomato-tpgsw"
# )

# version = project.version(4)

# dataset = version.download("yolov8")


# print("Dataset descargado en:", dataset.location)



# MODELO DE CLASIFICACIÓN DE TOMATE CHERRY
from roboflow import Roboflow


rf = Roboflow(api_key="API_ROBOTFLOW")


project = rf.workspace("cherrytomato-n6w1z").project(
    "cherry-tomato-classification-weatf"
)

version = project.version(2)

dataset = version.download("yolov8")


print("Dataset descargado en:", dataset.location)