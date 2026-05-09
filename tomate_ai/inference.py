import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class TomatoInference:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent

        # ==================================================
        # MODELO SEGMENTACIÓN
        # ==================================================
        self.segment_model = YOLO(
            str(
                BASE_DIR
                / "runs"
                / "segment"
                / "runs"
                / "segment"
                / "tomato_growth"
                / "weights"
                / "best.pt"
            )
        )

        # ==================================================
        # MODELO CRECIMIENTO
        # ==================================================
        self.growth_model = YOLO(
            str(
                BASE_DIR
                / "models"
                / "detector_best.pt"
            )
        )

        # ==================================================
        # CONFIGURACIÓN
        # ==================================================
        self.SEGMENT_CONFIDENCE = 0.35
        self.GROWTH_CONFIDENCE = 0.10

        # ==================================================
        # CLASES
        # ==================================================
        self.VALID_CLASSES = {
            "Vegetative Stage",
            "Flower Bud",
            "Anthesis",
            "Fruit Bud",
            "Fruit Maturation",
            "Fully Grown"
        }

        # ==================================================
        # DÍAS COSECHA
        # ==================================================
        self.harvest_map = {
            "Vegetative Stage": 60,
            "Flower Bud": 45,
            "Anthesis": 35,
            "Fruit Bud": 25,
            "Fruit Maturation": 12,
            "Fully Grown": 0
        }

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
    # MEDIDAS
    # ======================================================
    def _calculate_size(self, mask):

        mask = (mask * 255).astype(np.uint8)

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

        area = cv2.contourArea(largest)

        (_, _), radius = cv2.minEnclosingCircle(
            largest
        )

        diameter_px = radius * 2

        x, y, w, h = cv2.boundingRect(largest)

        return {
            "area_px": round(float(area), 2),
            "diametro_px": round(float(diameter_px), 2),
            "width_px": int(w),
            "height_px": int(h)
        }

    # ======================================================
    # IMAGEN ANOTADA
    # ======================================================
    def _generate_annotated_image(self, result):

        plotted = result.plot()

        _, buffer = cv2.imencode(
            ".jpg",
            plotted
        )

        return base64.b64encode(
            buffer
        ).decode("utf-8")

    # ======================================================
    # INFERENCIA
    # ======================================================
    def predict_base64(self, image_base64):

        image_path = self._decode_image(
            image_base64
        )

        try:

            # ==================================================
            # SEGMENTACIÓN
            # ==================================================
            segment_results = self.segment_model.predict(
                source=image_path,
                conf=self.SEGMENT_CONFIDENCE,
                retina_masks=True,
                verbose=False
            )

            result = segment_results[0]

            # ==================================================
            # VALIDAR
            # ==================================================
            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):
                return {
                    "success": False,
                    "message": "No se detectaron tomates."
                }

            if result.masks is None:
                return {
                    "success": False,
                    "message": "No se generaron máscaras."
                }

            # ==================================================
            # IMAGEN ANOTADA
            # ==================================================
            annotated_image = (
                self._generate_annotated_image(result)
            )

            # ==================================================
            # TOMATES DETECTADOS
            # ==================================================
            tomatoes = []

            total = len(result.boxes)

            for idx in range(total):

                box = result.boxes[idx]

                confidence = float(
                    box.conf[0]
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                mask = (
                    result.masks.data[idx]
                    .cpu()
                    .numpy()
                )

                size_data = self._calculate_size(
                    mask
                )

                tomato = {

                    "id": idx + 1,

                    "confianza": round(
                        confidence,
                        3
                    ),

                    "bounding_box": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    },

                    "medidas": size_data
                }

                tomatoes.append(tomato)

            # ==================================================
            # CRECIMIENTO
            # ==================================================
            growth_results = self.growth_model.predict(
                source=image_path,
                conf=self.GROWTH_CONFIDENCE,
                verbose=False
            )

            growth_result = growth_results[0]

            growth_data = []

            if (
                growth_result.boxes is not None
                and len(growth_result.boxes) > 0
            ):

                for box in growth_result.boxes:

                    class_id = int(box.cls[0])

                    class_name = (
                        growth_result.names[class_id]
                    )

                    confidence = float(box.conf[0])

                    if (
                        class_name
                        not in self.VALID_CLASSES
                    ):
                        continue

                    growth_data.append({

                        "estado": class_name,

                        "confianza": round(
                            confidence,
                            3
                        ),

                        "dias_cosecha": (
                            self.harvest_map.get(
                                class_name,
                                0
                            )
                        )
                    })

            # ==================================================
            # RESPUESTA
            # ==================================================
            return {

                "success": True,

                "total_tomates": len(tomatoes),

                "tomates": tomatoes,

                "etapas_detectadas": growth_data,

                "annotated_image": annotated_image
            }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

        finally:

            if os.path.exists(image_path):
                os.remove(image_path)