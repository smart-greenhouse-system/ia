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

        MONEDA_DIR = (
            BASE_DIR.parent
            / "moneda_ai"
        )

        # ==================================================
        # MODELO SEGMENTACIÓN TOMATES
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
        # MODELO CLASIFICACIÓN
        # ==================================================
        self.classify_model = YOLO(
            str(
                BASE_DIR
                / "runs"
                / "classify"
                / "tomato_classifier"
                / "weights"
                / "best.pt"
            )
        )

        # ==================================================
        # MODELO MONEDAS
        # ==================================================
        self.coin_model = YOLO(
            str(
                MONEDA_DIR
                / "runs"
                / "segment"
                / "runs"
                / "segment"
                / "coin_segmentation"
                / "weights"
                / "best.pt"
            )
        )

        # ==================================================
        # CONFIGURACIÓN
        # ==================================================
        self.SEGMENT_CONFIDENCE = 0.35
        self.GROWTH_CONFIDENCE = 0.15
        self.COIN_CONFIDENCE = 0.15

        # ==================================================
        # MONEDA REFERENCIA
        # ==================================================
        self.COIN_REAL_DIAMETER_MM = 20

        # ==================================================
        # ETAPAS
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
    # CALCULAR MEDIDAS
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
    # DETECTAR MONEDA
    # ======================================================
    def _detect_coin(self, image_path):

        results = self.coin_model.predict(

            source=image_path,

            conf=self.COIN_CONFIDENCE,

            retina_masks=True,

            imgsz=640,

            verbose=False
        )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return None

        if result.masks is None:
            return None

        best_idx = int(
            np.argmax(
                result.boxes.conf.cpu().numpy()
            )
        )

        mask = (
            result.masks.data[best_idx]
            .cpu()
            .numpy()
        )

        size_data = self._calculate_size(mask)

        if not size_data:
            return None

        return {

            "clase": "Moneda Colombiana",

            "confianza": round(
                float(
                    result.boxes.conf[best_idx]
                ),
                3
            ),

            "diametro_px":
                size_data["diametro_px"],

            "width_px":
                size_data["width_px"],

            "height_px":
                size_data["height_px"]
        }

    # ======================================================
    # PIXELES -> CM
    # ======================================================
    def _px_to_cm(
        self,
        px,
        coin_px
    ):

        if not coin_px:
            return None

        mm_per_px = (
            self.COIN_REAL_DIAMETER_MM
            / coin_px
        )

        mm = px * mm_per_px

        cm = mm / 10

        return round(cm, 2)

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
    # CLASIFICACIÓN
    # ======================================================
    def _classify_tomato(self, image_path):

        try:

            results = self.classify_model.predict(

                source=image_path,

                verbose=False
            )

            result = results[0]

            probs = result.probs

            if probs is None:
                return None

            class_id = int(probs.top1)

            confidence = float(probs.top1conf)

            class_name = result.names[class_id]

            return {

                "clase": class_name,

                "confianza": round(
                    confidence,
                    3
                )
            }

        except Exception as e:

            return {

                "clase": "Error",

                "confianza": 0,

                "error": str(e)
            }

    # ======================================================
    # INFERENCIA
    # ======================================================
    def predict_base64(self, image_base64):

        image_path = self._decode_image(
            image_base64
        )

        try:

            # ==================================================
            # MONEDA
            # ==================================================
            coin_data = self._detect_coin(
                image_path
            )

            coin_diameter_px = None

            if coin_data:
                coin_diameter_px = (
                    coin_data["diametro_px"]
                )

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

            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):
                return {

                    "success": False,

                    "message":
                        "No se detectaron tomates."
                }

            if result.masks is None:
                return {

                    "success": False,

                    "message":
                        "No se generaron máscaras."
                }

            annotated_image = (
                self._generate_annotated_image(
                    result
                )
            )

            # ==================================================
            # CLASIFICACIÓN
            # ==================================================
            classify_data = self._classify_tomato(
                image_path
            )

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

                    growth_data.append({

                        "estado": class_name,

                        "confianza": round(
                            confidence,
                            3
                        ),

                        "dias_cosecha":
                            self.harvest_map.get(
                                class_name,
                                0
                            )
                    })

            # ==================================================
            # TOMATES
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

                # ==============================================
                # MEDIDAS REALES
                # ==============================================
                real_data = {

                    "diametro_cm": None,

                    "ancho_cm": None,

                    "alto_cm": None
                }

                if (
                    size_data
                    and coin_diameter_px
                ):

                    real_data = {

                        "diametro_cm":
                            self._px_to_cm(
                                size_data["diametro_px"],
                                coin_diameter_px
                            ),

                        "ancho_cm":
                            self._px_to_cm(
                                size_data["width_px"],
                                coin_diameter_px
                            ),

                        "alto_cm":
                            self._px_to_cm(
                                size_data["height_px"],
                                coin_diameter_px
                            )
                    }

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

                    "medidas": size_data,

                    "medidas_reales": real_data
                }

                tomatoes.append(tomato)

            # ==================================================
            # RESPUESTA JSON
            # ==================================================
            return {

                "success": True,

                "clasificacion":
                    classify_data,

                "total_tomates":
                    len(tomatoes),

                "tomates":
                    tomatoes,

                "etapas_detectadas":
                    growth_data,

                "monedas":
                    [coin_data]
                    if coin_data else [],

                "annotated_image":
                    annotated_image
            }

        except Exception as e:

            return {

                "success": False,

                "message": str(e)
            }

        finally:

            if os.path.exists(image_path):
                os.remove(image_path)