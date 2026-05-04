import base64
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

        self.stage_info = {
            "Harvest Stage": {
                "apto_cosecha": True,
                "recomendacion": "Lechuga lista para cosechar."
            },
            "Vegetative Stage": {
                "apto_cosecha": False,
                "recomendacion": "Planta en desarrollo foliar. Aún no ha alcanzado el tamaño óptimo."
            },
            "Seedling Stage": {
                "apto_cosecha": False,
                "recomendacion": "Planta en etapa de plántula. Requiere más tiempo de desarrollo."
            }
        }

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
                    "message": "No se detectó planta de lechuga, vuelve a subir otra imagen"
                }

            box = results.boxes[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id]

            info = self.stage_info.get(class_name, {
                "apto_cosecha": False,
                "recomendacion": "Etapa desconocida"
            })

            return {
                "success": True,
                "etapa": class_name,
                "apto_cosecha": info["apto_cosecha"],
                "recomendacion": info["recomendacion"],
                "confianza": round(confidence, 3)
            }
        finally:
            os.remove(image_path)
