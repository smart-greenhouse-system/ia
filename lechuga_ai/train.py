from model import LechugaDetector

detector = LechugaDetector(weights="yolov8n.pt")

detector.train_model(
    data_yaml="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
