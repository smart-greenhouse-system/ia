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
        self.BASE_DIR = Path(__file__).resolve().parent

        # Dispositivo
        self.DEVICE = 0 if torch.cuda.is_available() else "cpu"
        print(f"\n🚀 DEVICE: {self.DEVICE}")

        # Rutas a modelos
        self.SEGMENT_MODEL_PATH = (
            self.BASE_DIR
            / "runs" / "segment" / "runs" / "segment"
            / "tomato_growth" / "weights" / "best.pt"
        )
        self.GROWTH_MODEL_PATH = (
            self.BASE_DIR / "models" / "best.pt"
        )

        self._validate_model(self.SEGMENT_MODEL_PATH)
        self._validate_model(self.GROWTH_MODEL_PATH)

        print("\n📦 Cargando modelos...")
        self.segment_model = YOLO(str(self.SEGMENT_MODEL_PATH))
        self.growth_model = YOLO(str(self.GROWTH_MODEL_PATH))
        print("✅ MODELOS CARGADOS")

        # Configuraciones de inferencia
        self.CONFIDENCE = 0.25
        self.IOU = 0.45
        self.IMAGE_SIZE = 960  # Segmentación
        self.GROWTH_IMG_SIZE = 640 # Crecimiento

        # Tamaño real de la carta (Ancho x Alto en cm)
        self.CARD_WIDTH_CM = 6.3
        self.CARD_HEIGHT_CM = 8.8
        self.SCALE_PX_PER_CM = 100.0  # Escala virtual rectificada (1cm = 100px)

        # Mapeo de cosechas y días
        self.harvest_map = {
            "vegetative_stage": {"dias": 60, "descripcion": "Etapa vegetativa"},
            "flower_bud": {"dias": 45, "descripcion": "Botón floral"},
            "anthesis": {"dias": 35, "descripcion": "Flor abierta"},
            "fruit_bud": {"dias": 25, "descripcion": "Fruto pequeño"},
            "fruit_maturation": {"dias": 12, "descripcion": "Maduración"},
            "fully_grown": {"dias": 0, "descripcion": "Listo para cosecha"}
        }

        # Colores visuales
        self.class_colors = {
            "vegetative_stage": (0, 180, 0),
            "flower_bud": (0, 255, 255),
            "anthesis": (0, 200, 255),
            "fruit_bud": (0, 140, 255),
            "fruit_maturation": (0, 80, 255),
            "fully_grown": (0, 0, 255),
            "card": (255, 255, 255)
        }

    def _validate_model(self, path):
        print(f"\n🔍 VALIDANDO:\n{path}")
        if not path.exists():
            raise FileNotFoundError(f"\n❌ Modelo no encontrado:\n{path}")
        print("✅ MODELO OK")

    def _decode_image(self, image_base64):
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp.write(image_bytes)
            temp.close()
            return temp.name
        except Exception as e:
            raise Exception(f"Error decodificando imagen: {e}")

    def _clean_mask(self, mask):
        mask = (mask > 0.5).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _get_homography_from_card(self, result):
        """
        Encuentra la máscara de la carta, detecta sus 4 esquinas 
        y genera la matriz de transformación de perspectiva (Homografía).
        """
        try:
            if result.boxes is None or result.masks is None:
                return None, None

            best_card_mask = None
            best_conf = 0
            
            # Buscar la mejor detección de la carta
            for idx, box in enumerate(result.boxes):
                class_name = result.names[int(box.cls[0])].lower().strip()
                if class_name != "card":
                    continue
                
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_card_mask = result.masks.data[idx].cpu().numpy()

            if best_card_mask is None:
                return None, None

            clean_mask = self._clean_mask(best_card_mask)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None

            contour = max(contours, key=cv2.contourArea)

            # Aproximar el contorno a un polígono (buscamos 4 esquinas)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Si no obtiene 4 esquinas limpias, forzamos un rectángulo mínimo
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(contour)
                pts = cv2.boxPoints(rect).astype(np.float32)

            # Ordenar puntos: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            rect_pts = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect_pts[0] = pts[np.argmin(s)]
            rect_pts[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect_pts[1] = pts[np.argmin(diff)]
            rect_pts[3] = pts[np.argmax(diff)]

            # Determinar si la carta está en vertical u horizontal
            widthA = np.linalg.norm(rect_pts[2] - rect_pts[3])
            widthB = np.linalg.norm(rect_pts[1] - rect_pts[0])
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(rect_pts[2] - rect_pts[1])
            heightB = np.linalg.norm(rect_pts[3] - rect_pts[0])
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth > maxHeight:
                real_w, real_h = self.CARD_HEIGHT_CM, self.CARD_WIDTH_CM # Landscape
            else:
                real_w, real_h = self.CARD_WIDTH_CM, self.CARD_HEIGHT_CM # Portrait

            # Mapeo a nuestro plano corregido (1cm = 100px)
            dst_w = real_w * self.SCALE_PX_PER_CM
            dst_h = real_h * self.SCALE_PX_PER_CM

            dst_pts = np.array([
                [0, 0],
                [dst_w, 0],
                [dst_w, dst_h],
                [0, dst_h]
            ], dtype="float32")

            # Matriz de homografía
            H = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            
            return H, best_card_mask

        except Exception as e:
            print(f"❌ HOMOGRAPHY ERROR: {e}")
            return None, None

    def _calculate_real_metrics(self, mask, H):
        """
        Proyecta la máscara del tomate en el plano corregido por perspectiva 
        y calcula su área real en cm².
        """
        try:
            clean_mask = self._clean_mask(mask)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None

            contour = max(contours, key=cv2.contourArea)

            if H is not None:
                # Proyectar el contorno usando la matriz de perspectiva
                contour_float = contour.astype(np.float32)
                warped_contour = cv2.perspectiveTransform(contour_float, H)
                
                # Area en el plano corregido (pixeles virtuales)
                area_warped_px = cv2.contourArea(warped_contour)
                
                # Convertir a cm2
                area_cm2 = area_warped_px / (self.SCALE_PX_PER_CM ** 2)
                
                # Calcular medidas adicionales rectificadas
                (_, _), radius_warped = cv2.minEnclosingCircle(warped_contour)
                diametro_cm = (radius_warped * 2) / self.SCALE_PX_PER_CM
                
                x, y, w, h = cv2.boundingRect(warped_contour)
                ancho_cm = w / self.SCALE_PX_PER_CM
                alto_cm = h / self.SCALE_PX_PER_CM
                
            else:
                # Fallback por si la carta no se detectó (sin escala real)
                return None

            return {
                "area_cm2": round(area_cm2, 2),
                "diametro_cm": round(diametro_cm, 2),
                "ancho_cm": round(ancho_cm, 2),
                "alto_cm": round(alto_cm, 2)
            }

        except Exception as e:
            print(f"❌ REAL SIZE ERROR: {e}")
            return None

    def _classify_growth_stage(self, image_path):
        results = self.growth_model.predict(
            source=image_path,
            conf=0.20,
            iou=0.45,
            imgsz=self.GROWTH_IMG_SIZE,
            device=self.DEVICE,
            verbose=False
        )
        detections = []
        result = results[0]

        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            class_name = names[int(box.cls[0])].lower().strip()
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": xyxy
            })
        return detections

    def _get_growth_data(self, class_name, confidence):
        data = self.harvest_map.get(
            class_name,
            {"dias": None, "descripcion": "Desconocido"}
        )
        return {
            "estado": class_name,
            "confianza": round(confidence, 3),
            "dias_cosecha": data["dias"],
            "descripcion": data["descripcion"]
        }

    def _draw_segmentation(self, image, mask, label, box, class_name):
        clean_mask = self._clean_mask(mask)
        color = self.class_colors.get(class_name, (0, 255, 0))
        
        overlay = image.copy()
        overlay[clean_mask > 0] = color
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label_bg_y = max(y1 - 30, 0)
        # Ajuste dinámico del fondo del texto para etiquetas largas
        cv2.rectangle(image, (x1, label_bg_y), (x1 + 320, y1), color, -1)
        
        cv2.putText(
            image, label, (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
        )
        return image

    def predict_base64(self, image_base64):
        image_path = self._decode_image(image_base64)

        try:
            print("\n🚀 INICIANDO PREDICCIÓN")

            # 1. Segmentación Base (Detecta tomates y la carta)
            results = self.segment_model.predict(
                source=image_path,
                conf=self.CONFIDENCE,
                iou=self.IOU,
                retina_masks=True,
                imgsz=self.IMAGE_SIZE,
                device=self.DEVICE,
                verbose=False
            )

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return {"success": False, "message": "No hay detecciones en la imagen."}

            original = cv2.imread(image_path)
            annotated = original.copy()

            # 2. Generar Homografía desde la Carta
            H, card_mask = self._get_homography_from_card(result)
            carta_detectada = H is not None

            # VALIDACIÓN 1: ¿Se detectó la carta?
            if not carta_detectada:
                return {
                    "success": False, 
                    "message": "Error: No se detectó la carta de referencia (6.3 x 8.8 cm). Es obligatoria para calcular las medidas reales."
                }

            # 3. Dibujar la carta
            for idx, box in enumerate(result.boxes):
                class_name = result.names[int(box.cls[0])].lower().strip()
                if class_name == "card":
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    annotated = self._draw_segmentation(
                        annotated, card_mask, "CARTA REFERENCIA", xyxy, "card"
                    )
                    break

            # 4. Obtener Etapas de Crecimiento
            growth_results = self._classify_growth_stage(image_path)

            # 5. Procesar Tomates
            tomatoes = []
            tomato_id = 1

            for idx in range(len(result.boxes)):
                box = result.boxes[idx]
                class_name = result.names[int(box.cls[0])].lower().strip()

                if class_name == "card":
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                t_area = (x2 - x1) * (y2 - y1)
                
                mask = result.masks.data[idx].cpu().numpy()

                # -- Intersección robusta con el modelo de crecimiento --
                growth_stage = class_name
                growth_conf = float(box.conf[0])
                best_overlap = 0.0

                for g in growth_results:
                    gx1, gy1, gx2, gy2 = g["box"]
                    inter_x1 = max(x1, gx1)
                    inter_y1 = max(y1, gy1)
                    inter_x2 = min(x2, gx2)
                    inter_y2 = min(y2, gy2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        overlap = inter_area / float(t_area)
                        
                        # Si cubre más del 30% del tomate, asumimos que es el mismo
                        if overlap > best_overlap and overlap > 0.3:
                            best_overlap = overlap
                            growth_stage = g["class"]
                            growth_conf = g["confidence"]

                growth_data = self._get_growth_data(growth_stage, growth_conf)

                # -- Calcular medidas reales con Homografía --
                real_data = self._calculate_real_metrics(mask, H)

                if real_data:
                    label = (
                        f"{growth_stage.upper()} | "
                        f"{real_data['diametro_cm']}cm | "
                        f"{real_data['area_cm2']}cm2"
                    )
                else:
                    label = f"{growth_stage.upper()} | Escala Fallida"
                    real_data = {
                        "diametro_cm": None, "ancho_cm": None,
                        "alto_cm": None, "area_cm2": None
                    }

                annotated = self._draw_segmentation(
                    annotated, mask, label, (x1, y1, x2, y2), growth_stage
                )

                tomatoes.append({
                    "id": tomato_id,
                    "estado": growth_stage,
                    "confianza": round(growth_conf, 3),
                    "dias_cosecha": growth_data["dias_cosecha"],
                    "descripcion": growth_data["descripcion"],
                    "medidas_reales": real_data
                })
                tomato_id += 1

            # VALIDACIÓN 2: ¿Se detectaron tomates?
            if len(tomatoes) == 0:
                return {
                    "success": False,
                    "message": "Error: Se detectó la carta, pero no se encontraron tomates en la imagen."
                }

            # Generar imagen final
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_image = base64.b64encode(buffer).decode("utf-8")

            return {
                "success": True,
                "total_tomates": len(tomatoes),
                "carta_detectada": carta_detectada,
                "tomates": tomatoes,
                "annotated_image": annotated_image
            }

        except Exception as e:
            print(f"\n❌ GLOBAL ERROR: {e}")
            return {"success": False, "message": str(e)}

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)