import torch.nn as nn
from ultralytics import YOLO

class LechugaDetector(nn.Module):

    def __init__(self, weights="yolov8n-seg.pt"):
        super().__init__()
        self.model = YOLO(weights)

    def forward(self, x):
        return self.model(x)

    def train_model(self, data_yaml, epochs=50, imgsz=640, batch=16):
        return self.model.train(
            data=data_yaml,
            task="segment",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            save=True,
            save_period=1
        )

    def predict(self, source, conf=0.25):
        return self.model(source, conf=conf)

    def save(self, path):
        self.model.save(path)
