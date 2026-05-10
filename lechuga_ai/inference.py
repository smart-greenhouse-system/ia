import base64
import cv2
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import LechugaDetector

BASE_DIR = Path(__file__).parent


class LechugaInference:

    def __init__(self):
        self.detector = LechugaDetector(weights=str(BASE_DIR / "models" / "best.pt"))

    def _decode_image(self, image_base64):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(image_bytes)
        temp.close()

        return temp.name

    def _encode_image(self, img_array):
        _, buffer = cv2.imencode(".jpg", img_array)
        return base64.b64encode(buffer).decode("utf-8")

    def predict_base64(self, image_base64):
        image_path = self._decode_image(image_base64)

        try:
            results = self.detector.predict(image_path)[0]

            if results.boxes is None or len(results.boxes) == 0:
                return {
                    "success": False,
                    "message": "No se detectó planta de lechuga, vuelve a subir otra imagen"
                }

            box = results.boxes[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id]

            tiene_mascara = results.masks is not None and len(results.masks) > 0

            annotated = results.plot()
            annotated_b64 = self._encode_image(annotated)

            return {
                "success": True,
                "etapa": class_name,
                "confianza": round(confidence, 3),
                "tiene_mascara": tiene_mascara,
                "annotated_image": annotated_b64
            }
        finally:
            os.remove(image_path)
