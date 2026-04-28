import base64
import tempfile
from ultralytics import YOLO


class TomatoInference:

    def __init__(self):
        self.model = YOLO("models/best.pt")

        # lógica agrícola 
        self.harvest_map = {
            "Vegetative Stage": 60,
            "Flower Bud": 45,
            "Anthesis": 35,
            "Fruit Bud": 25,
            "Fruit Maturation": 12,
            "Fully Grown": 0
        }

    
    # BASE64 -> ARCHIVO
    def _decode_image(self, image_base64):
        # eliminar data:image/jpeg;base64,XXXX
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(image_bytes)
        temp.close()

        return temp.name

    # =========================
    # PREDICCIÓN PRINCIPAL
    # =========================
    def predict_base64(self, image_base64):

        image_path = self._decode_image(image_base64)

        results = self.model(image_path)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return {
                "success": False,
                "message": "No se detectó planta de tomate, vuelve a subir otra imagen"
                
            }

        box = results.boxes[0]

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = results.names[class_id]

        days = self.harvest_map.get(class_name, "Desconocido")

        return {
            "success": True,
            "estado_planta": class_name,
            "confianza": round(confidence, 3),
            "tiempo_cosecha_dias": days
        }