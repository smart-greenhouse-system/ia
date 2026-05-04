import base64
import tempfile
import os
from ultralytics import YOLO


class TomatoInference:
    """
    Sistema de inferencia de doble etapa:

    1. Detector:
       Verifica si realmente existe una planta de tomate cherry.

    2. Clasificador:
       Determina la etapa de crecimiento.

    Esto evita falsos positivos como:
    - Melones
    - Pepinos
    - Otras plantas
    """

    def __init__(self):
        # ==========================================
        # MODELO 1: Detector de tomate cherry
        # ==========================================
        self.detector = YOLO("models/best.pt")

        # ==========================================
        # MODELO 2: Clasificador de etapas
        # ==========================================
        self.growth_model = YOLO("models/detector_best.pt")

        # Umbral mínimo para aceptar detecciones
        self.MIN_CONFIDENCE = 0.75

        # Clases válidas del modelo de crecimiento
        self.VALID_CLASSES = {
            "Vegetative Stage",
            "Flower Bud",
            "Anthesis",
            "Fruit Bud",
            "Fruit Maturation",
            "Fully Grown"
        }

        # Días estimados restantes para cosecha
        self.harvest_map = {
            "Vegetative Stage": 60,
            "Flower Bud": 45,
            "Anthesis": 35,
            "Fruit Bud": 25,
            "Fruit Maturation": 12,
            "Fully Grown": 0
        }

    def _decode_image(self, image_base64):
        """
        Convierte una imagen Base64 en un archivo temporal.
        """

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

    def predict_base64(self, image_base64):
        """
        Flujo completo de inferencia:
        1. Detectar si es tomate cherry.
        2. Clasificar etapa de crecimiento.
        """

        image_path = self._decode_image(image_base64)

        try:
            # ==================================================
            # PASO 1: DETECCIÓN DE TOMATE CHERRY
            # ==================================================
            detection_results = self.detector(image_path)[0]

            if detection_results.boxes is None or len(detection_results.boxes) == 0:
                return {
                    "success": False,
                    "message": (
                        "No se detectó ninguna planta de tomate cherry "
                        "en la imagen."
                    )
                }

            # Mejor detección del detector
            best_detection = max(
                detection_results.boxes,
                key=lambda box: float(box.conf[0])
            )

            detection_confidence = float(best_detection.conf[0])

            if detection_confidence < self.MIN_CONFIDENCE:
                return {
                    "success": False,
                    "message": (
                        "La imagen no parece contener una planta "
                        "de tomate cherry."
                    ),
                    "confianza_detector": round(
                        detection_confidence, 3
                    )
                }

            # ==================================================
            # PASO 2: CLASIFICACIÓN DE ETAPA
            # ==================================================
            growth_results = self.growth_model(image_path)[0]

            if growth_results.boxes is None or len(growth_results.boxes) == 0:
                return {
                    "success": False,
                    "message": (
                        "Se detectó tomate cherry, pero no fue posible "
                        "determinar la etapa de crecimiento."
                    )
                }

            # Mejor predicción del clasificador
            best_box = max(
                growth_results.boxes,
                key=lambda box: float(box.conf[0])
            )

            class_id = int(best_box.cls[0])
            confidence = float(best_box.conf[0])
            class_name = growth_results.names[class_id]

            # Validar confianza del clasificador
            if confidence < self.MIN_CONFIDENCE:
                return {
                    "success": False,
                    "message": (
                        "La etapa de crecimiento no pudo determinarse "
                        "con suficiente confianza."
                    ),
                    "confianza_clasificador": round(
                        confidence, 3
                    )
                }

            # Validar clase
            if class_name not in self.VALID_CLASSES:
                return {
                    "success": False,
                    "message": "Clase detectada no válida.",
                    "clase_detectada": class_name
                }

            # Calcular días restantes
            days = self.harvest_map[class_name]

            # ==================================================
            # RESPUESTA EXITOSA
            # ==================================================
            return {
                "success": True,
                "tomate_detectado": True,
                "estado_planta": class_name,
                "confianza_detector": round(
                    detection_confidence, 3
                ),
                "confianza_clasificador": round(
                    confidence, 3
                ),
                "tiempo_cosecha_dias": days
            }

        finally:
            # Eliminar archivo temporal
            if os.path.exists(image_path):
                os.remove(image_path)