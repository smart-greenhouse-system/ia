import base64
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from moneda_ai.model import MonedaDetector

BASE_DIR = Path(__file__).parent


class MonedaInference:

    def __init__(self):

        self.detector = MonedaDetector(
            weights=str(
                BASE_DIR
                / "models"
                / "best.pt"
            )
        )

        # ==========================================
        # TAMAÑO REAL MONEDA
        # Moneda colombiana 100 pesos
        # diámetro aprox 2.1 cm
        # ==========================================
        self.COIN_DIAMETER_CM = 2.1

    # ======================================================
    # BASE64 -> IMAGEN
    # ======================================================
    def _decode_image(self, image_base64):

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp.write(image_bytes)
        temp.close()

        return temp.name

    # ======================================================
    # CALCULAR TAMAÑO
    # ======================================================
    def _calculate_size(self, mask):

        mask = (
            mask * 255
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest
        )

        (_, _), radius = (
            cv2.minEnclosingCircle(
                largest
            )
        )

        diameter_px = radius * 2

        x, y, w, h = cv2.boundingRect(
            largest
        )

        return {

            "area_px": round(
                float(area),
                2
            ),

            "diametro_px": round(
                float(diameter_px),
                2
            ),

            "width_px": int(w),

            "height_px": int(h)
        }

    # ======================================================
    # INFERENCIA
    # ======================================================
    def predict_base64(self, image_base64):

        image_path = self._decode_image(
            image_base64
        )

        try:

            results = self.detector.predict(
                image_path
            )[0]

            # ==================================================
            # VALIDAR DETECCIONES
            # ==================================================
            if (
                results.boxes is None
                or len(results.boxes) == 0
            ):
                return {

                    "success": False,

                    "message":
                    "No se detectaron monedas"
                }

            if results.masks is None:
                return {

                    "success": False,

                    "message":
                    "El modelo no generó máscaras"
                }

            monedas = []

            # ==================================================
            # PROCESAR TODAS
            # ==================================================
            for i, box in enumerate(results.boxes):

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    results.names[class_id]
                )

                # ==========================================
                # MÁSCARA
                # ==========================================
                mask = (
                    results.masks.data[i]
                    .cpu()
                    .numpy()
                )

                size_data = (
                    self._calculate_size(
                        mask
                    )
                )

                # ==========================================
                # ESCALA REAL
                # ==========================================
                cm_per_pixel = None

                if size_data:

                    cm_per_pixel = round(
                        self.COIN_DIAMETER_CM
                        / size_data["diametro_px"],
                        5
                    )

                moneda = {

                    "clase": class_name,

                    "confianza": round(
                        confidence,
                        3
                    ),

                    "medidas": size_data,

                    "escala": {

                        "cm_por_pixel":
                        cm_per_pixel,

                        "diametro_real_cm":
                        self.COIN_DIAMETER_CM
                    }
                }

                monedas.append(moneda)

            # ==================================================
            # RESPUESTA
            # ==================================================
            return {

                "success": True,

                "total_monedas":
                len(monedas),

                "monedas":
                monedas
            }

        except Exception as e:

            return {

                "success": False,

                "message": str(e)
            }

        finally:

            if os.path.exists(image_path):
                os.remove(image_path)