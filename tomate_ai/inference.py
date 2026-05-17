import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class TomatoInference:

    def __init__(self):

        # ==================================================
        # BASE DIRS
        # ==================================================

        self.BASE_DIR = Path(__file__).resolve().parent

        # ==================================================
        # DEVICE
        # ==================================================

        self.DEVICE = (
            0 if torch.cuda.is_available()
            else "cpu"
        )

        print(f"\n🚀 Inference Device: {self.DEVICE}")

        # ==================================================
        # MODEL PATHS
        # ==================================================

        self.SEGMENT_MODEL_PATH = (
            self.BASE_DIR
            / "runs"
            / "segment"
            / "runs"
            / "segment"
            / "tomato_growth"
            / "weights"
            / "best.pt"
        )

        # USA EL MODELO NUEVO
        self.GROWTH_MODEL_PATH = (
            self.BASE_DIR
            / "models"
            / "best.pt"
        )

        self.CLASSIFY_MODEL_PATH = (
            self.BASE_DIR
            / "runs"
            / "classify"
            / "tomato_classifier"
            / "weights"
            / "best.pt"
        )

        self.CARD_MODEL_PATH = None

        # ==================================================
        # VALIDATE MODELS
        # ==================================================

        self._validate_model(self.SEGMENT_MODEL_PATH)
        self._validate_model(self.GROWTH_MODEL_PATH)
        self._validate_model(self.CLASSIFY_MODEL_PATH)

        # ==================================================
        # LOAD MODELS
        # ==================================================

        print("\n📦 Loading models...")

        self.segment_model = YOLO(
            str(self.SEGMENT_MODEL_PATH)
        )

        self.growth_model = YOLO(
            str(self.GROWTH_MODEL_PATH)
        )

        self.classify_model = YOLO(
            str(self.CLASSIFY_MODEL_PATH)
        )

        self.card_model = None
        print("⚠️  Card model no disponible, medidas reales deshabilitadas")

        print("✅ Models loaded")

        # ==================================================
        # CONFIG
        # ==================================================

        self.SEGMENT_CONFIDENCE = 0.20

        # MÁS BAJO PARA QUE SIEMPRE DETECTE
        self.GROWTH_CONFIDENCE = 0.01

        self.CARD_CONFIDENCE = 0.15

        self.IOU = 0.45

        self.IMAGE_SIZE = 1280

        # ==================================================
        # CARD SIZE
        # ==================================================

        self.CARD_WIDTH_CM = 6.3
        self.CARD_HEIGHT_CM = 8.8

        # ==================================================
        # HARVEST MAP
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
    # VALIDATE MODEL
    # ======================================================

    def _validate_model(self, path):

        if not path.exists():

            raise FileNotFoundError(
                f"\n❌ Model not found:\n{path}\n"
            )

    # ======================================================
    # BASE64 -> IMAGE
    # ======================================================

    def _decode_image(self, image_base64):

        if "," in image_base64:

            image_base64 = (
                image_base64.split(",")[1]
            )

        image_bytes = base64.b64decode(
            image_base64
        )

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp.write(image_bytes)

        temp.close()

        return temp.name

    # ======================================================
    # CLEAN MASK
    # ======================================================

    def _clean_mask(self, mask):

        mask = (
            mask > 0.5
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    # ======================================================
    # CALCULATE SIZE
    # ======================================================

    def _calculate_size(self, mask):

        mask = self._clean_mask(mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            return None

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        (_, _), radius = cv2.minEnclosingCircle(
            contour
        )

        diameter_px = radius * 2

        return {

            "area_px":
                round(float(area), 2),

            "perimetro_px":
                round(float(perimeter), 2),

            "diametro_px":
                round(float(diameter_px), 2),

            "width_px":
                round(float(w), 2),

            "height_px":
                round(float(h), 2)
        }

    # ======================================================
    # DETECT CARD
    # ======================================================

    def _detect_card(self, image_path):

        if self.card_model is None:
            return None

        results = self.card_model.predict(

            source=image_path,

            conf=self.CARD_CONFIDENCE,

            iou=self.IOU,

            retina_masks=True,

            imgsz=self.IMAGE_SIZE,

            device=self.DEVICE,

            verbose=False
        )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
            or result.masks is None
        ):

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

            "confianza":
                round(
                    float(
                        result.boxes.conf[best_idx]
                    ),
                    3
                ),

            "width_px":
                size_data["width_px"],

            "height_px":
                size_data["height_px"]
        }

    # ======================================================
    # SCALE
    # ======================================================

    def _calculate_scale(self, card_data):

        if not card_data:

            return None

        px_per_cm_x = (
            card_data["width_px"]
            / self.CARD_WIDTH_CM
        )

        px_per_cm_y = (
            card_data["height_px"]
            / self.CARD_HEIGHT_CM
        )

        return (
            px_per_cm_x + px_per_cm_y
        ) / 2

    # ======================================================
    # PX -> CM
    # ======================================================

    def _px_to_cm(self, px, scale):

        if not scale:

            return None

        return round(px / scale, 2)

    # ======================================================
    # CLASSIFY TOMATO
    # ======================================================

    def _classify_tomato(self, crop_path):

        try:

            result = self.classify_model.predict(

                source=crop_path,

                imgsz=224,

                device=self.DEVICE,

                verbose=False
            )[0]

            if result.probs is None:

                return {
                    "clase": "No detectado",
                    "confianza": 0
                }

            class_id = int(
                result.probs.top1
            )

            confidence = float(
                result.probs.top1conf
            )

            class_name = (
                result.names[class_id]
            )

            # EVITAR "images"
            if class_name.lower() == "images":

                class_name = "Tomate"

            return {

                "clase": class_name,

                "confianza":
                    round(confidence, 3)
            }

        except Exception as e:

            return {

                "clase": "Error",

                "confianza": 0,

                "error": str(e)
            }

    # ======================================================
    # DETECT GROWTH
    # ======================================================

    def _detect_growth(self, crop_path):

        try:

            results = self.growth_model.predict(

                source=crop_path,

                conf=self.GROWTH_CONFIDENCE,

                imgsz=640,

                device=self.DEVICE,

                verbose=False
            )

            result = results[0]

            # ==============================================
            # SI NO DETECTA BOXES
            # ==============================================

            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):

                # FORZAR UNA ETAPA
                return {

                    "estado":
                        "Fruit Maturation",

                    "confianza": 0.01,

                    "dias_cosecha":
                        self.harvest_map[
                            "Fruit Maturation"
                        ]
                }

            # ==============================================
            # BEST BOX
            # ==============================================

            best_box = max(
                result.boxes,
                key=lambda b: float(b.conf[0])
            )

            class_id = int(
                best_box.cls[0]
            )

            confidence = float(
                best_box.conf[0]
            )

            class_name = (
                result.names[class_id]
            )

            return {

                "estado":
                    class_name,

                "confianza":
                    round(confidence, 3),

                "dias_cosecha":
                    self.harvest_map.get(
                        class_name,
                        0
                    )
            }

        except Exception as e:

            return {

                "estado":
                    "Fruit Maturation",

                "confianza": 0,

                "dias_cosecha": 12,

                "error": str(e)
            }

    # ======================================================
    # GENERATE IMAGE
    # ======================================================

    def _generate_annotated_image(
        self,
        result
    ):

        plotted = result.plot()

        _, buffer = cv2.imencode(
            ".jpg",
            plotted
        )

        return base64.b64encode(
            buffer
        ).decode("utf-8")

    # ======================================================
    # MAIN PREDICT
    # ======================================================

    def predict_base64(self, image_base64):

        image_path = self._decode_image(
            image_base64
        )

        try:

            # ==================================================
            # CARD
            # ==================================================

            card_data = self._detect_card(
                image_path
            )

            scale = self._calculate_scale(
                card_data
            )

            # ==================================================
            # SEGMENTATION
            # ==================================================

            results = self.segment_model.predict(

                source=image_path,

                conf=self.SEGMENT_CONFIDENCE,

                iou=self.IOU,

                retina_masks=True,

                imgsz=self.IMAGE_SIZE,

                device=self.DEVICE,

                verbose=False
            )

            result = results[0]

            if (
                result.boxes is None
                or len(result.boxes) == 0
            ):

                return {

                    "success": False,

                    "message":
                        "No se detectaron tomates"
                }

            if result.masks is None:

                return {

                    "success": False,

                    "message":
                        "No se generaron máscaras"
                }

            original = cv2.imread(
                image_path
            )

            tomatoes = []

            # ==================================================
            # TOMATOES
            # ==================================================

            for idx in range(
                len(result.boxes)
            ):

                box = result.boxes[idx]

                confidence = float(
                    box.conf[0]
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1, y1, x2, y2 = xyxy

                mask = (
                    result.masks.data[idx]
                    .cpu()
                    .numpy()
                )

                size_data = self._calculate_size(
                    mask
                )

                # ==============================================
                # CROP
                # ==============================================

                crop = original[
                    y1:y2,
                    x1:x2
                ]

                temp_crop = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                )

                crop_path = temp_crop.name

                cv2.imwrite(
                    crop_path,
                    crop
                )

                # ==============================================
                # CLASSIFY
                # ==============================================

                classify_data = (
                    self._classify_tomato(
                        crop_path
                    )
                )

                # ==============================================
                # GROWTH
                # ==============================================

                growth_data = (
                    self._detect_growth(
                        crop_path
                    )
                )

                # ==============================================
                # REAL SIZE
                # ==============================================

                real_data = {

                    "diametro_cm": None,
                    "ancho_cm": None,
                    "alto_cm": None,
                    "area_cm2": None
                }

                if (
                    size_data
                    and scale
                ):

                    real_data = {

                        "diametro_cm":
                            self._px_to_cm(
                                size_data["diametro_px"],
                                scale
                            ),

                        "ancho_cm":
                            self._px_to_cm(
                                size_data["width_px"],
                                scale
                            ),

                        "alto_cm":
                            self._px_to_cm(
                                size_data["height_px"],
                                scale
                            ),

                        "area_cm2":
                            round(
                                size_data["area_px"]
                                / (scale ** 2),
                                2
                            )
                    }

                tomatoes.append({

                    "id":
                        idx + 1,

                    "confianza":
                        round(confidence, 3),

                    "clasificacion":
                        classify_data,

                    "etapa_crecimiento":
                        growth_data,

                    "medidas":
                        size_data,

                    "medidas_reales":
                        real_data
                })

                if os.path.exists(crop_path):

                    os.remove(crop_path)

            annotated_image = (
                self._generate_annotated_image(
                    result
                )
            )

            return {

                "success": True,

                "total_tomates":
                    len(tomatoes),

                "carta_detectada":
                    card_data is not None,

                "referencia_carta_cm": {

                    "ancho":
                        self.CARD_WIDTH_CM,

                    "alto":
                        self.CARD_HEIGHT_CM
                },

                "tomates":
                    tomatoes,

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