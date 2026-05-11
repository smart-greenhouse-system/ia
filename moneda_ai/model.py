import torch.nn as nn
from ultralytics import YOLO


class MonedaDetector(nn.Module):

    def __init__(self, weights="yolov8n-seg.pt"):
        super().__init__()
        self.model = YOLO(weights)
        self.last_train_path = None  

    def forward(self, x):
        return self.model(x)

    def train_model(
        self,
        data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/segment",
        name="moneda_modelo"
    ):
        results = self.model.train(
            data=data_yaml,
            task="segment",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            save=True,
            save_period=1
        )

        # 👇 guardamos la ruta del entrenamiento
        self.last_train_path = f"{project}/{name}"
        return results

    def predict(self, source, conf=0.5):
        return self.model(source, conf=conf)

    def load_best(self):
        if self.last_train_path:
            self.model = YOLO(f"{self.last_train_path}/weights/best.pt")
        else:
            raise ValueError("Primero entrena el modelo")

    def load_last(self):
        if self.last_train_path:
            self.model = YOLO(f"{self.last_train_path}/weights/last.pt")
        else:
            raise ValueError("Primero entrena el modelo")