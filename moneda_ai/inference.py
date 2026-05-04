import base64
import os
import tempfile

from model import MonedaDetector


class MonedaInference:

    def __init__(self):
        self.detector = MonedaDetector(weights="models/best.pt")

    def _decode_image(self, image_base64):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(image_bytes)
        temp.close()

        return temp.name

    def predict_base64(self, image_base64):
        image_path = self._decode_image(image_base64)

        try:
            results = self.detector.predict(image_path)[0]

            if results.boxes is None or len(results.boxes) == 0:
                return {
                    "success": False,
                    "message": "No se detectaron monedas en la imagen"
                }

            monedas = []
            for i, box in enumerate(results.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results.names[class_id]

                moneda = {
                    "clase": class_name,
                    "confianza": round(confidence, 3),
                    "tiene_mascara": results.masks is not None and i < len(results.masks)
                }
                monedas.append(moneda)

            return {
                "success": True,
                "total_monedas": len(monedas),
                "monedas": monedas
            }
        finally:
            os.remove(image_path)
