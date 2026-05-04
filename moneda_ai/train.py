from model import MonedaDetector

detector = MonedaDetector(weights="yolov8n-seg.pt")

detector.train_model(
    data_yaml="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
