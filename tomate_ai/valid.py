from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO(
    "/home/pablo/Documents/Python/crecimiento/tomate_ai/models/best.pt"
)

metrics = model.val(
    data="/home/pablo/Documents/Python/crecimiento/tomate_ai/dataset/data.yaml"
)

# ejemplo simple
valores = [
    metrics.box.map50,
    metrics.box.map,
    metrics.box.mp,
    metrics.box.mr
]

nombres = ["mAP50", "mAP50-95", "Precision", "Recall"]

plt.bar(nombres, valores)
plt.ylabel("Valor")
plt.title("Métricas del modelo YOLOv8")
plt.show()