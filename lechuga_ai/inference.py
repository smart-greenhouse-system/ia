import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from model import LechugaDetector

BASE_DIR = Path(__file__).parent

AREA_CARD_CM2 = 6.3 * 8.8  # 55.44 cm²
ID_CLASE_TARJETA = 3


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

    def _encode_annotated(self, results):
        imagen_bgr = results.plot()
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(imagen_rgb)
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG", quality=85)
        b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def predict_base64(self, image_base64):
        image_path = self._decode_image(image_base64)

        try:
            results = self.detector.predict(image_path, conf=0.5)[0]

            imagen_base64 = self._encode_annotated(results)

            if results.masks is None or results.boxes is None or len(results.boxes) == 0:
                return {
                    "success": False,
                    "message": "No se detectó ningún objeto en la imagen.",
                    "imagen_base64": imagen_base64
                }

            clases = results.boxes.cls.cpu().numpy()
            mascaras = results.masks.data.cpu().numpy()
            nombres = results.names

            pixeles_tarjeta = 0
            pixeles_lechuga = 0
            etapa_detectada = "No detectada"

            for idx, clase_id in enumerate(clases):
                clase_id_int = int(clase_id)
                nombre = nombres[clase_id_int].lower()

                if clase_id_int == ID_CLASE_TARJETA or "card" in nombre or "tarjeta" in nombre:
                    pixeles_tarjeta += int(np.sum(mascaras[idx] > 0))
                else:
                    pixeles_lechuga += int(np.sum(mascaras[idx] > 0))
                    etapa_detectada = nombres[clase_id_int]

            if pixeles_tarjeta == 0:
                return {
                    "success": False,
                    "message": "No se detectó la tarjeta de referencia.",
                    "imagen_base64": imagen_base64
                }

            if pixeles_lechuga == 0:
                return {
                    "success": False,
                    "message": "No se detectó la lechuga.",
                    "imagen_base64": imagen_base64
                }

            factor = AREA_CARD_CM2 / pixeles_tarjeta
            area_real_cm2 = round(pixeles_lechuga * factor, 2)

            return {
                "success": True,
                "etapa_crecimiento": etapa_detectada,
                "area_foliar_cm2": area_real_cm2,
                "imagen_base64": imagen_base64,
                "metadatos": {
                    "resolucion_cm2_por_pixel": round(factor, 6)
                }
            }

        finally:
            os.remove(image_path)
