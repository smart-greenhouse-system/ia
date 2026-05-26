import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ================================================================
# HELPER: Convertir tipos numpy → Python nativos para JSON
# ESTA ES LA CAUSA RAÍZ del "❌ JSON ERROR": np.float32, np.int64
# etc. no son serializables por json.dumps() de Python.
# ================================================================

def _to_json_safe(obj: Any) -> Any:
    """Convierte recursivamente tipos numpy a tipos Python nativos."""

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj


class TomatoInference:

    def __init__(self) -> None:

        self.BASE_DIR = Path(__file__).resolve().parent

        # ==================================================
        # DEVICE
        # ==================================================

        self.DEVICE = 0 if torch.cuda.is_available() else "cpu"
        print(f"\n🚀 [INIT] DEVICE: {self.DEVICE}")

        # ==================================================
        # MODELOS
        # ==================================================

        self.SEGMENT_MODEL_PATH = Path(
            "/home/pablo/Documents/Python/crecimiento/"
            "tomate_ai/runs/segment/runs/segment/"
            "tomate_model_v2-2/weights/best.pt"
        )

        self.GROWTH_MODEL_PATH = Path(
            "/home/pablo/Documents/Python/crecimiento/"
            "tomate_ai/models/best.pt"
        )

        self._validate_model(self.SEGMENT_MODEL_PATH, "Segmentación")
        self._validate_model(self.GROWTH_MODEL_PATH,  "Crecimiento")

        print("\n📦 [INIT] Cargando modelos...")
        self.segment_model = YOLO(str(self.SEGMENT_MODEL_PATH))
        self.growth_model  = YOLO(str(self.GROWTH_MODEL_PATH))
        print("✅ [INIT] Ambos modelos cargados correctamente")

        # ==================================================
        # CONFIGURACIÓN BASE
        # ==================================================

        self.CONFIDENCE      = 0.30   # Reducido para mayor sensibilidad
        self.CONFIDENCE_MIN  = 0.10   # Mínimo absoluto en reintentos
        self.IOU             = 0.45
        self.IMAGE_SIZE      = 960
        self.GROWTH_IMG_SIZE = 640

        # ==================================================
        # CARTA REFERENCIA
        # ==================================================

        self.CARD_WIDTH_CM   = 6.3
        self.CARD_HEIGHT_CM  = 8.8
        self.SCALE_PX_PER_CM = 100.0

        # ==================================================
        # CLASES TOMATE
        # ==================================================

        self.TOMATO_CLASSES = [
            "vegetative_stage",
            "flower_bud",
            "anthesis",
            "fruit_bud",
            "fruit_maturation",
            "fully_grown",
        ]

        # ==================================================
        # MAPA DE ETAPAS — días y descripciones
        # dias siempre es int (nunca None)
        # ==================================================

        self.harvest_map = {
            "vegetative_stage": {
                "dias":        75,
                "descripcion": (
                    "Etapa vegetativa: la planta desarrolla "
                    "tallos, hojas y raíces. Aún no hay flores."
                ),
            },
            "flower_bud": {
                "dias":        55,
                "descripcion": (
                    "Botón floral: aparecen los primeros "
                    "brotes florales antes de la antesis."
                ),
            },
            "anthesis": {
                "dias":        45,
                "descripcion": (
                    "Antesis: flor completamente abierta "
                    "y en proceso de polinización."
                ),
            },
            "fruit_bud": {
                "dias":        30,
                "descripcion": (
                    "Cuaje de fruto: fruto pequeño recién "
                    "formado tras la polinización exitosa."
                ),
            },
            "fruit_maturation": {
                "dias":        14,
                "descripcion": (
                    "Maduración: el fruto cambia de color "
                    "y acumula azúcares. Próximo a cosechar."
                ),
            },
            "fully_grown": {
                "dias":        0,
                "descripcion": (
                    "Listo para cosecha: tomate maduro, "
                    "color y firmeza óptimos para recolección."
                ),
            },
            "_unknown": {
                "dias":        -1,
                "descripcion": (
                    "Etapa no reconocida por el modelo. "
                    "Revisa la imagen o el modelo entrenado."
                ),
            },
        }

        # ==================================================
        # COLORES BGR por clase
        # ==================================================

        self.class_colors = {
            "vegetative_stage": (0,   180,   0),
            "flower_bud":       (0,   255, 255),
            "anthesis":         (0,   200, 255),
            "fruit_bud":        (0,   140, 255),
            "fruit_maturation": (0,    80, 255),
            "fully_grown":      (0,     0, 255),
            "card":             (255, 255, 255),
        }

    # ===========================================================
    # VALIDAR MODELO
    # ===========================================================

    def _validate_model(self, path: Path, name: str = "") -> None:
        label = f"[MODELO {name}]" if name else "[MODELO]"
        print(f"\n🔍 {label} Validando: {path}")
        if not path.exists():
            print(f"❌ {label} Archivo NO encontrado")
            raise FileNotFoundError(f"{label} No encontrado: {path}")
        print(f"✅ {label} OK")

    # ===========================================================
    # DECODIFICAR IMAGEN
    # ===========================================================

    def _decode_image(self, image_base64: str) -> str:

        print("\n🔍 [DECODE] Decodificando imagen base64...")

        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]

            image_bytes = base64.b64decode(image_base64)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(image_bytes)
            tmp.close()

            print(f"✅ [DECODE] Imagen guardada en: {tmp.name}")
            return tmp.name

        except Exception as exc:
            print(f"❌ [DECODE] Error: {exc}")
            raise RuntimeError(f"Error decodificando imagen: {exc}") from exc

    # ===========================================================
    # CALIDAD DE IMAGEN — heurística rápida
    # Retorna dict con blur_score y brightness para elegir
    # la estrategia de preprocesado adecuada.
    # ===========================================================

    def _assess_quality(self, img: np.ndarray) -> dict:
        """Evalúa blur (Laplacian variance) y luminosidad media."""

        gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness  = float(gray.mean())

        print(f"📊 [CALIDAD] Blur score={blur_score:.1f} | "
              f"Brillo medio={brightness:.1f}")

        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "is_blurry":  blur_score < 80.0,
            "is_dark":    brightness < 60.0,
            "is_bright":  brightness > 200.0,
        }

    # ===========================================================
    # ESTRATEGIAS DE PREPROCESADO
    # Genera múltiples versiones de la imagen para maximizar
    # la probabilidad de detección sin importar la calidad.
    # ===========================================================

    def _preprocess_clahe_sharpen(self, img: np.ndarray) -> np.ndarray:
        """CLAHE + sharpening — mejora general."""
        kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                             dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        lab       = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b   = cv2.split(lab)
        clahe     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l         = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _preprocess_denoise(self, img: np.ndarray) -> np.ndarray:
        """Denoising — para imágenes ruidosas / granuladas."""
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h=10, hColor=10,
            templateWindowSize=7, searchWindowSize=21
        )
        return self._preprocess_clahe_sharpen(denoised)

    def _preprocess_brightness(self, img: np.ndarray) -> np.ndarray:
        """Corrección de brillo + CLAHE — para oscuras / sobreexpuestas."""
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _preprocess_gamma(
        self, img: np.ndarray, gamma: float = 1.4
    ) -> np.ndarray:
        """Corrección gamma — saca detalles en sombras."""
        inv_gamma = 1.0 / gamma
        table     = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(img, table)

    def _preprocess_bilateral(self, img: np.ndarray) -> np.ndarray:
        """Filtro bilateral — suaviza ruido conservando bordes."""
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return self._preprocess_clahe_sharpen(smooth)

    def _build_preprocessing_pipeline(
        self, img: np.ndarray, quality: dict
    ) -> list[tuple[str, np.ndarray]]:
        """
        Construye una lista priorizada de estrategias según la
        calidad de la imagen. Cada tupla es (nombre, imagen_procesada).
        """

        pipeline: list[tuple[str, np.ndarray]] = []

        # 1. CLAHE + sharpen — siempre incluido primero
        pipeline.append(("clahe_sharpen", self._preprocess_clahe_sharpen(img)))

        # 2. Ajustar por condición detectada
        if quality["is_blurry"]:
            print("🔧 [PIPELINE] Imagen borrosa detectada → "
                  "añadiendo bilateral + denoising")
            pipeline.append(("bilateral", self._preprocess_bilateral(img)))
            pipeline.append(("denoise",   self._preprocess_denoise(img)))

        if quality["is_dark"]:
            print("🔧 [PIPELINE] Imagen oscura detectada → "
                  "añadiendo gamma + brightness")
            pipeline.append(("gamma",      self._preprocess_gamma(img, 1.6)))
            pipeline.append(("brightness", self._preprocess_brightness(img)))

        if quality["is_bright"]:
            print("🔧 [PIPELINE] Sobreexposición detectada → "
                  "añadiendo gamma oscurecido")
            pipeline.append(("gamma_dark", self._preprocess_gamma(img, 0.7)))

        # 3. Original como último fallback
        pipeline.append(("original", img.copy()))

        print(f"✅ [PIPELINE] {len(pipeline)} estrategia(s) preparadas: "
              f"{[n for n, _ in pipeline]}")

        return pipeline

    # ===========================================================
    # GUARDAR IMAGEN TEMPORAL
    # ===========================================================

    def _save_temp(self, img: np.ndarray) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp.name, img)
        tmp.close()
        return tmp.name

    # ===========================================================
    # EJECUTAR SEGMENTACIÓN CON REINTENTOS MULTI-ESTRATEGIA
    # ===========================================================

    def _run_segmentation(
        self,
        pipeline: list[tuple[str, np.ndarray]],
        original_path: str
    ) -> tuple[Any, str]:
        """
        Intenta la segmentación con cada estrategia del pipeline.
        Devuelve (result, nombre_estrategia_usada).
        Garantiza que siempre retorna algo (nunca None).
        """

        confidences = [
            self.CONFIDENCE,
            self.CONFIDENCE - 0.05,
            self.CONFIDENCE - 0.10,
            max(self.CONFIDENCE - 0.15, self.CONFIDENCE_MIN),
        ]

        best_result   = None
        best_strategy = "ninguna"
        best_count    = 0

        for strategy_name, img_array in pipeline:

            tmp_path = self._save_temp(img_array)

            try:
                for conf in confidences:

                    conf = float(max(conf, self.CONFIDENCE_MIN))

                    print(f"🔎 [SEGMENT] Estrategia='{strategy_name}' "
                          f"conf={conf:.2f}...")

                    results = self.segment_model.predict(
                        source=tmp_path,
                        conf=conf,
                        iou=self.IOU,
                        retina_masks=True,
                        imgsz=self.IMAGE_SIZE,
                        device=self.DEVICE,
                        verbose=False,
                    )

                    result = results[0]
                    n_boxes = (
                        len(result.boxes)
                        if result.boxes is not None
                        else 0
                    )

                    if n_boxes > 0:
                        print(f"✅ [SEGMENT] '{strategy_name}' → "
                              f"{n_boxes} objeto(s) detectado(s) "
                              f"con conf={conf:.2f}")

                        # Guardamos el resultado con más detecciones
                        if n_boxes > best_count:
                            best_count    = n_boxes
                            best_result   = result
                            best_strategy = strategy_name

                        # Si encontramos algo razonable, no seguir bajando conf
                        break

            except Exception as exc:
                print(f"❌ [SEGMENT] Error en estrategia "
                      f"'{strategy_name}': {exc}")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if best_result is None or best_count == 0:
            print("❌ [SEGMENT] Ninguna estrategia produjo detecciones")
        else:
            print(f"✅ [SEGMENT] Mejor estrategia: '{best_strategy}' "
                  f"con {best_count} objeto(s)")

        return best_result, best_strategy

    # ===========================================================
    # LIMPIAR MÁSCARA
    # ===========================================================

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask   = (mask > 0.5).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ===========================================================
    # VALIDAR TOMATE
    # ===========================================================

    def _is_valid_tomato(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
        box_area: int,
    ) -> bool:

        try:
            if box_area < 800:
                print(f"⚠️ [VALIDAR] Caja muy pequeña ({box_area}px²)")
                return False

            clean_mask = self._clean_mask(mask)
            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("⚠️ [VALIDAR] Sin contornos")
                return False

            contour      = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour)

            if contour_area < 400:
                print(f"⚠️ [VALIDAR] Área contorno pequeña "
                      f"({contour_area:.0f}px²)")
                return False

            aspect_ratio = width / float(max(height, 1))
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                print(f"⚠️ [VALIDAR] Aspect ratio inválido "
                      f"({aspect_ratio:.2f})")
                return False

            hull      = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area <= 0:
                print("⚠️ [VALIDAR] Hull area = 0")
                return False

            solidity = contour_area / hull_area
            if solidity < 0.45:
                print(f"⚠️ [VALIDAR] Solidez baja ({solidity:.2f})")
                return False

            return True

        except Exception as exc:
            print(f"❌ [VALIDAR] Error: {exc}")
            return False

    # ===========================================================
    # DATOS ETAPA — nunca retorna None ni vacío
    # ===========================================================

    def _get_growth_data(
        self,
        class_name: str,
        confidence: float,
    ) -> dict:

        data = self.harvest_map.get(class_name)

        if data is None:
            print(f"⚠️ [GROWTH_DATA] Clase desconocida '{class_name}' "
                  "→ usando fallback '_unknown'")
            data = self.harvest_map["_unknown"]

        dias        = int(data["dias"])
        descripcion = str(data["descripcion"])

        if dias > 0:
            dias_texto = f"Aproximadamente {dias} días para la cosecha"
        elif dias == 0:
            dias_texto = "Listo para cosechar hoy"
        else:
            dias_texto = "No calculable (etapa no reconocida)"

        print(f"✅ [GROWTH_DATA] '{class_name}' | "
              f"Días={dias} | Conf={confidence:.3f}")

        return {
            "estado":       str(class_name),
            "confianza":    round(float(confidence), 3),
            "dias_cosecha": dias,
            "dias_texto":   dias_texto,
            "descripcion":  descripcion,
        }

    # ===========================================================
    # ESCALA SIN CARTA
    # ===========================================================

    def _estimate_scale_without_card(
        self, image_shape: tuple
    ) -> float:
        px_per_cm = float(image_shape[0]) / 30.0
        print(f"⚠️ [ESCALA] Sin carta → estimación {px_per_cm:.2f} px/cm")
        return px_per_cm

    # ===========================================================
    # MEDIDAS EN PÍXELES (sin carta)
    # TODOS los valores se convierten a Python nativos → JSON-safe
    # ===========================================================

    def _calculate_pixel_metrics(
        self,
        mask: np.ndarray,
        image_shape: tuple,
    ) -> Optional[dict]:

        print("\n📏 [MEDIDAS_PX] Calculando medidas en píxeles...")

        try:
            clean_mask = self._clean_mask(mask)
            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("❌ [MEDIDAS_PX] Sin contornos")
                return None

            contour = max(contours, key=cv2.contourArea)

            (_, _), radius_px = cv2.minEnclosingCircle(contour)
            x, y, w, h        = cv2.boundingRect(contour)
            area_px           = cv2.contourArea(contour)
            px_per_cm         = self._estimate_scale_without_card(image_shape)

            # ── Conversión explícita a float Python ─────────────
            radius_px = float(radius_px)
            w         = int(w)
            h_box     = int(h)
            area_px   = float(area_px)
            px_per_cm = float(px_per_cm)

            diametro_px = round(radius_px * 2, 1)
            diametro_cm = round((radius_px * 2) / px_per_cm, 2)
            ancho_cm    = round(w / px_per_cm, 2)
            alto_cm     = round(h_box / px_per_cm, 2)
            area_cm2    = round(area_px / (px_per_cm ** 2), 2)

            result = {
                "diametro_px": float(diametro_px),
                "ancho_px":    w,
                "alto_px":     h_box,
                "area_px2":    float(area_px),
                "diametro_cm": float(diametro_cm),
                "ancho_cm":    float(ancho_cm),
                "alto_cm":     float(alto_cm),
                "area_cm2":    float(area_cm2),
                "escala_real": False,
            }

            print(f"✅ [MEDIDAS_PX] Ø={diametro_px}px "
                  f"(~{diametro_cm}cm) | Área={area_px:.0f}px²")

            return result

        except Exception as exc:
            print(f"❌ [MEDIDAS_PX] Error: {exc}")
            return None

    # ===========================================================
    # HOMOGRAFÍA CARTA
    # ===========================================================

    def _get_homography_from_card(
        self, result: Any
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        print("\n🃏 [CARTA] Buscando carta de referencia...")

        try:
            if result is None or result.boxes is None or result.masks is None:
                print("⚠️ [CARTA] Sin cajas/máscaras disponibles")
                return None, None

            best_card_idx = None
            best_conf     = 0.0

            for idx, box in enumerate(result.boxes):
                class_name = (
                    result.names[int(box.cls[0])].lower().strip()
                )
                if class_name != "card":
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf     = conf
                    best_card_idx = idx

            if best_card_idx is None:
                print("⚠️ [CARTA] Ninguna carta detectada")
                return None, None

            print(f"✅ [CARTA] Carta encontrada (idx={best_card_idx}, "
                  f"conf={best_conf:.3f})")

            card_mask  = result.masks.data[best_card_idx].cpu().numpy()
            clean_mask = self._clean_mask(card_mask)

            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("❌ [CARTA] Sin contornos en máscara de carta")
                return None, card_mask

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                print("✅ [CARTA] Polígono de 4 vértices encontrado")
            else:
                rect = cv2.minAreaRect(contour)
                pts  = cv2.boxPoints(rect).astype(np.float32)
                print(f"⚠️ [CARTA] {len(approx)} vértices → "
                      "usando minAreaRect")

            # Ordenar: TL, TR, BR, BL
            rect_pts    = np.zeros((4, 2), dtype="float32")
            s           = pts.sum(axis=1)
            rect_pts[0] = pts[np.argmin(s)]
            rect_pts[2] = pts[np.argmax(s)]
            diff        = np.diff(pts, axis=1)
            rect_pts[1] = pts[np.argmin(diff)]
            rect_pts[3] = pts[np.argmax(diff)]

            wA       = np.linalg.norm(rect_pts[2] - rect_pts[3])
            wB       = np.linalg.norm(rect_pts[1] - rect_pts[0])
            hA       = np.linalg.norm(rect_pts[2] - rect_pts[1])
            hB       = np.linalg.norm(rect_pts[3] - rect_pts[0])
            maxWidth  = max(int(wA), int(wB))
            maxHeight = max(int(hA), int(hB))

            if maxWidth > maxHeight:
                real_w, real_h = self.CARD_HEIGHT_CM, self.CARD_WIDTH_CM
            else:
                real_w, real_h = self.CARD_WIDTH_CM, self.CARD_HEIGHT_CM

            dst_w = real_w * self.SCALE_PX_PER_CM
            dst_h = real_h * self.SCALE_PX_PER_CM

            dst_pts = np.array([
                [0,     0    ],
                [dst_w, 0    ],
                [dst_w, dst_h],
                [0,     dst_h],
            ], dtype="float32")

            H = cv2.getPerspectiveTransform(rect_pts, dst_pts)

            print(f"✅ [CARTA] Homografía calculada | "
                  f"{real_w}cm × {real_h}cm")

            return H, card_mask

        except Exception as exc:
            print(f"❌ [CARTA] Error: {exc}")
            return None, None

    # ===========================================================
    # MEDIDAS REALES CON CARTA
    # TODOS los valores se convierten a Python nativos → JSON-safe
    # ===========================================================

    def _calculate_real_metrics(
        self,
        mask: np.ndarray,
        H: np.ndarray,
    ) -> Optional[dict]:

        print("\n📐 [MEDIDAS_REALES] Calculando con homografía...")

        try:
            if H is None:
                print("❌ [MEDIDAS_REALES] H es None")
                return None

            clean_mask = self._clean_mask(mask)
            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("❌ [MEDIDAS_REALES] Sin contornos")
                return None

            contour        = max(contours, key=cv2.contourArea)
            contour_float  = contour.astype(np.float32)
            warped_contour = cv2.perspectiveTransform(contour_float, H)

            area_warped_px    = float(cv2.contourArea(warped_contour))
            area_cm2          = area_warped_px / (self.SCALE_PX_PER_CM ** 2)

            (_, _), radius_w  = cv2.minEnclosingCircle(warped_contour)
            diametro_cm       = (float(radius_w) * 2) / self.SCALE_PX_PER_CM

            x, y, w, h        = cv2.boundingRect(warped_contour)
            ancho_cm          = float(w) / self.SCALE_PX_PER_CM
            alto_cm           = float(h) / self.SCALE_PX_PER_CM

            result = {
                "area_cm2":    round(float(area_cm2),    2),
                "diametro_cm": round(float(diametro_cm), 2),
                "ancho_cm":    round(float(ancho_cm),    2),
                "alto_cm":     round(float(alto_cm),     2),
                "escala_real": True,
            }

            print(f"✅ [MEDIDAS_REALES] Ø={result['diametro_cm']}cm | "
                  f"{result['ancho_cm']}×{result['alto_cm']}cm | "
                  f"Área={result['area_cm2']}cm²")

            return result

        except Exception as exc:
            print(f"❌ [MEDIDAS_REALES] Error: {exc}")
            return None

    # ===========================================================
    # CLASIFICACIÓN CRECIMIENTO (modelo 2)
    # ===========================================================

    def _normalize_growth_class(self, class_name: str) -> str:
        """
        Normaliza nombres de clases para que coincidan con harvest_map.
        Convierte nombres con espacios (como los que puede emitir el
        modelo) a los nombres con guión bajo usados internamente.
        """
        class_name = class_name.lower().strip()
        replacements = {
            "fully grown":       "fully_grown",
            "fruit maturation":  "fruit_maturation",
            "fruit bud":         "fruit_bud",
            "flower bud":        "flower_bud",
            "vegetative stage":  "vegetative_stage",
        }
        return replacements.get(class_name, class_name)

    def _classify_growth_stage(self, image_path: str) -> list:

        print("\n🌱 [GROWTH_MODEL] Ejecutando modelo de crecimiento...")
        detections: list = []

        try:
            results = self.growth_model.predict(
                source=image_path,
                conf=0.20,
                iou=0.45,
                imgsz=self.GROWTH_IMG_SIZE,
                device=self.DEVICE,
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:
                print("⚠️ [GROWTH_MODEL] Sin detecciones")
                return detections

            for box in result.boxes:
                raw_class_name = (
                    result.names[int(box.cls[0])].lower().strip()
                )

                # ==================================================
                # NORMALIZACIÓN DE NOMBRES
                # ==================================================
                class_name = self._normalize_growth_class(raw_class_name)

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                detections.append({
                    "class":      class_name,
                    "confidence": conf,
                    "box":        xyxy,
                })

                print(
                    f"   📦 [GROWTH_MODEL] "
                    f"raw='{raw_class_name}' "
                    f"→ normalized='{class_name}' "
                    f"conf={conf:.3f}"
                )

            print(f"✅ [GROWTH_MODEL] {len(detections)} detecciones")

        except Exception as exc:
            print(f"❌ [GROWTH_MODEL] Error: {exc}")

        return detections

    # ===========================================================
    # DIBUJAR SEGMENTACIÓN
    # ===========================================================

    def _draw_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        label: str,
        box: tuple,
        class_name: str,
    ) -> np.ndarray:
        """
        Dibuja máscara, bounding-box y etiqueta de dos líneas sobre la
        imagen.  Divide 'label' en dos partes usando '|' como separador:
          línea 1 → nombre del estado  (ej. FULLY_GROWN)
          línea 2 → medidas            (ej. Ø 3.98 cm | 9.17 cm²)
        Si no hay '|' en el label, toda la cadena va en la línea 1.
        La caja de texto se pinta siempre dentro de los límites de la imagen.
        """

        h_img, w_img = image.shape[:2]

        # ── 1. Máscara: redimensionar al tamaño real de la imagen ──────
        clean_mask = self._clean_mask(mask)
        if clean_mask.shape[:2] != (h_img, w_img):
            clean_mask = cv2.resize(
                clean_mask, (w_img, h_img),
                interpolation=cv2.INTER_NEAREST
            )

        # ── 2. Overlay de color ─────────────────────────────────────────
        color   = self.class_colors.get(class_name, (0, 255, 0))
        overlay = image.copy()
        overlay[clean_mask > 0] = color
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

        # ── 3. Bounding-box ─────────────────────────────────────────────
        x1, y1, x2, y2 = (
            int(box[0]), int(box[1]), int(box[2]), int(box[3])
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # ── 4. Dividir label en dos líneas ──────────────────────────────
        parts = label.split("|", 1)
        line1 = parts[0].strip()
        line2 = parts[1].strip() if len(parts) > 1 else ""

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness  = 1
        pad        = 4          # padding interno px
        line_gap   = 4          # espacio entre líneas px

        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness) \
                      if line2 else ((0, 0), 0)

        box_w = max(w1, w2) + pad * 2
        box_h = h1 + (h2 + line_gap if line2 else 0) + pad * 2

        # Intentar colocar el label ENCIMA del bounding-box; si no cabe,
        # colocarlo DENTRO (esquina superior izquierda de la caja).
        if y1 - box_h - 2 >= 0:
            bg_y1 = y1 - box_h - 2
            bg_y2 = y1 - 2
        else:
            bg_y1 = y1 + 2
            bg_y2 = y1 + box_h + 2

        # Ajustar horizontalmente para no salirse a la derecha
        bg_x1 = x1
        bg_x2 = x1 + box_w
        if bg_x2 > w_img:
            bg_x1 = max(0, w_img - box_w)
            bg_x2 = w_img

        # ── 5. Fondo del label ──────────────────────────────────────────
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # ── 6. Texto línea 1 ────────────────────────────────────────────
        text_x  = bg_x1 + pad
        text_y1 = bg_y1 + pad + h1
        cv2.putText(
            image, line1,
            (text_x, text_y1),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

        # ── 7. Texto línea 2 (solo si existe) ──────────────────────────
        if line2:
            text_y2 = text_y1 + h2 + line_gap
            cv2.putText(
                image, line2,
                (text_x, text_y2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )

        return image

    # ===========================================================
    # PREDICCIÓN PRINCIPAL
    # ===========================================================

    def predict_base64(self, image_base64: str) -> dict:

        image_path = self._decode_image(image_base64)
        tmp_files: list[str] = [image_path]

        try:
            print("\n" + "=" * 60)
            print("🚀 [PREDICT] INICIO DE PREDICCIÓN")
            print("=" * 60)

            original = cv2.imread(image_path)

            if original is None:
                print("❌ [PREDICT] No se pudo leer la imagen")
                return self._error_response("Error al leer la imagen.")

            h_img, w_img = original.shape[:2]
            print(f"✅ [PREDICT] Imagen cargada: {w_img}×{h_img}px")

            # ──────────────────────────────────────────────────
            # PASO 1 — Evaluar calidad y construir pipeline
            # ──────────────────────────────────────────────────

            quality  = self._assess_quality(original)
            pipeline = self._build_preprocessing_pipeline(original, quality)

            # ──────────────────────────────────────────────────
            # PASO 2 — Segmentación multi-estrategia (modelo 1)
            # ──────────────────────────────────────────────────

            result, used_strategy = self._run_segmentation(
                pipeline, image_path
            )

            annotated = original.copy()

            if result is None or result.boxes is None or len(result.boxes) == 0:
                print("❌ [PREDICT] Sin detecciones tras todos los intentos")
                return {
                    **self._error_response(
                        "No se detectaron objetos. "
                        "Intenta acercar la cámara o mejorar la iluminación."
                    ),
                    "carta_detectada": False,
                    "aviso_carta":     None,
                    "total_tomates":   0,
                    "tomates":         [],
                }

            n_detected = len(result.boxes)
            print(f"✅ [PREDICT] {n_detected} objeto(s) detectado(s) "
                  f"con estrategia '{used_strategy}'")

            # ──────────────────────────────────────────────────
            # PASO 3 — Carta de referencia
            # ──────────────────────────────────────────────────

            H, card_mask    = self._get_homography_from_card(result)
            carta_detectada = H is not None
            aviso_carta: Optional[str] = None

            if carta_detectada:
                print("✅ [CARTA] Medidas en cm reales activadas")
            else:
                aviso_carta = (
                    "No se detectó la carta de referencia. "
                    "Las medidas se muestran en píxeles con estimación "
                    "aproximada en cm (baja confianza). "
                    "Para medidas precisas incluye la carta en la imagen."
                )
                print(f"⚠️ [CARTA] {aviso_carta}")

            # ──────────────────────────────────────────────────
            # PASO 4 — Dibujar carta
            # ──────────────────────────────────────────────────

            if carta_detectada and card_mask is not None:
                for idx, box in enumerate(result.boxes):
                    cn = result.names[int(box.cls[0])].lower().strip()
                    if cn != "card":
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    annotated = self._draw_segmentation(
                        annotated, card_mask,
                        "CARTA REFERENCIA", xyxy, "card"
                    )
                    print("✅ [DIBUJO] Carta dibujada")
                    break

            # ──────────────────────────────────────────────────
            # PASO 5 — Modelo de crecimiento (modelo 2)
            # Usa la mejor imagen del pipeline para mayor precisión
            # ──────────────────────────────────────────────────

            best_img_for_growth = pipeline[0][1]   # primera estrategia
            tmp_growth = self._save_temp(best_img_for_growth)
            tmp_files.append(tmp_growth)

            growth_results = self._classify_growth_stage(tmp_growth)

            # ──────────────────────────────────────────────────
            # PASO 6 — Procesar tomates
            # ──────────────────────────────────────────────────

            print(f"\n🍅 [TOMATES] Procesando {n_detected} detecciones...")

            tomatoes:  list[dict] = []
            tomato_id: int        = 1

            for idx in range(n_detected):

                try:
                    box        = result.boxes[idx]
                    class_name = (
                        result.names[int(box.cls[0])].lower().strip()
                    )

                    if class_name == "card":
                        continue

                    if class_name not in self.TOMATO_CLASSES:
                        print(f"⚠️ [TOMATE] Clase '{class_name}' ignorada")
                        continue

                    print(f"\n   🍅 [TOMATE {tomato_id}] "
                          f"Clase base='{class_name}'")

                    xyxy           = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), \
                                     int(xyxy[2]), int(xyxy[3])
                    width  = x2 - x1
                    height = y2 - y1
                    t_area = max(width * height, 1)

                    if result.masks is None:
                        print(f"⚠️ [TOMATE {tomato_id}] Sin máscara")
                        continue

                    mask = result.masks.data[idx].cpu().numpy()

                    # Validación geométrica
                    if not self._is_valid_tomato(mask, width, height, t_area):
                        print(f"⚠️ [TOMATE {tomato_id}] "
                              "Falso positivo; descartado")
                        continue

                    print(f"✅ [TOMATE {tomato_id}] Validación geométrica OK")

                    # Match con modelo de crecimiento
                    growth_stage = class_name
                    growth_conf  = float(box.conf[0])
                    best_overlap = 0.0

                    for g in growth_results:
                        gx1, gy1, gx2, gy2 = (
                            int(g["box"][0]), int(g["box"][1]),
                            int(g["box"][2]), int(g["box"][3])
                        )
                        ix1 = max(x1, gx1)
                        iy1 = max(y1, gy1)
                        ix2 = min(x2, gx2)
                        iy2 = min(y2, gy2)

                        if ix2 > ix1 and iy2 > iy1:
                            inter   = (ix2 - ix1) * (iy2 - iy1)
                            overlap = inter / float(t_area)
                            if overlap > best_overlap and overlap > 0.25:
                                best_overlap = overlap
                                growth_stage = g["class"]
                                growth_conf  = float(g["confidence"])

                    if best_overlap > 0.25:
                        print(f"✅ [TOMATE {tomato_id}] Refinado por "
                              f"modelo 2: '{growth_stage}' "
                              f"(overlap={best_overlap:.2f})")
                    else:
                        print(f"⚠️ [TOMATE {tomato_id}] Sin match modelo 2; "
                              f"usando modelo 1: '{growth_stage}'")

                    # Datos de crecimiento
                    growth_data = self._get_growth_data(
                        growth_stage, growth_conf
                    )

                    # Medidas
                    if carta_detectada:
                        real_data = self._calculate_real_metrics(mask, H)
                        if real_data is None:
                            print(f"⚠️ [TOMATE {tomato_id}] "
                                  "Fallback a medidas en píxeles")
                            real_data = self._calculate_pixel_metrics(
                                mask, original.shape
                            )
                    else:
                        real_data = self._calculate_pixel_metrics(
                            mask, original.shape
                        )

                    # ── Label en imagen ───────────────────────────────
                    # Línea 1: ID + estado  (siempre presente)
                    # Línea 2: medidas      (separada por el primer '|')
                    # _draw_segmentation divide por el primer '|'
                    estado_str = growth_stage.upper().replace("_", " ")
                    linea1 = f"#{tomato_id} {estado_str}"

                    if real_data is not None:
                        if real_data.get("escala_real", False):
                            linea2 = (
                                f"D:{real_data['diametro_cm']}cm "
                                f"A:{real_data['area_cm2']}cm2"
                            )
                        else:
                            linea2 = (
                                f"D:{real_data['diametro_px']}px "
                                f"(~{real_data['diametro_cm']}cm)"
                            )
                        label = f"{linea1} | {linea2}"
                    else:
                        label = f"{linea1} | SIN MEDIDAS"
                        print(f"⚠️ [TOMATE {tomato_id}] No se calcularon "
                              "medidas")

                    annotated = self._draw_segmentation(
                        annotated, mask, label,
                        (x1, y1, x2, y2), growth_stage
                    )

                    print(f"✅ [TOMATE {tomato_id}] Dibujado → {label}")

                    # Asegurar JSON-safe con _to_json_safe
                    tomato_entry = {
                        "id":             tomato_id,
                        "estado":         str(growth_data["estado"]),
                        "confianza":      float(growth_data["confianza"]),
                        "dias_cosecha":   int(growth_data["dias_cosecha"]),
                        "dias_texto":     str(growth_data["dias_texto"]),
                        "descripcion":    str(growth_data["descripcion"]),
                        "medidas_reales": _to_json_safe(real_data),
                    }

                    tomatoes.append(tomato_entry)

                    print(f"✅ [TOMATE {tomato_id}] Registrado → "
                          f"estado='{growth_data['estado']}' | "
                          f"días={growth_data['dias_cosecha']} | "
                          f"{growth_data['dias_texto']}")

                    tomato_id += 1

                except Exception as exc:
                    print(f"❌ [TOMATE idx={idx}] Error inesperado: {exc}")

            # ──────────────────────────────────────────────────
            # PASO 7 — Imagen anotada
            # (nunca imprimir el base64 en consola)
            # ──────────────────────────────────────────────────

            print("\n🖼️ [IMAGEN] Codificando imagen anotada...")

            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            print(f"✅ [IMAGEN] Codificada ({len(annotated_b64)} chars)")

            # ──────────────────────────────────────────────────
            # PASO 8 — Construir respuesta final JSON-safe
            # ──────────────────────────────────────────────────

            total         = len(tomatoes)
            tomato_error  = (
                None if total > 0 else
                "No se detectaron tomates válidos. "
                "Verifica que haya tomates visibles en la imagen."
            )

            if tomato_error:
                print(f"⚠️ [PREDICT] {tomato_error}")

            success = total > 0

            print("\n" + "=" * 60)
            print(f"{'✅' if success else '⚠️'} [PREDICT] FINALIZADO | "
                  f"Carta={'SÍ' if carta_detectada else 'NO'} | "
                  f"Estrategia='{used_strategy}' | "
                  f"Tomates={total}")
            print("=" * 60 + "\n")

            # _to_json_safe sobre toda la respuesta como seguro final
            response = _to_json_safe({
                "success":          success,
                "message":          "Predicción completada",
                "estrategia_usada": used_strategy,
                "errores": {
                    "card_error":   aviso_carta,
                    "tomato_error": tomato_error,
                },
                "carta_detectada":  carta_detectada,
                "aviso_carta":      aviso_carta,
                "total_tomates":    total,
                "tomates":          tomatoes,
                "annotated_image":  annotated_b64,
            })

            return response

        except Exception as exc:
            print(f"\n❌ [PREDICT] ERROR GLOBAL: {exc}")
            return self._error_response(str(exc))

        finally:
            for path in tmp_files:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"🗑️ [CLEANUP] Eliminado: {path}")
                    except Exception as exc:
                        print(f"⚠️ [CLEANUP] No se pudo eliminar "
                              f"{path}: {exc}")

    # ===========================================================
    # RESPUESTA DE ERROR ESTÁNDAR
    # ===========================================================

    @staticmethod
    def _error_response(message: str) -> dict:
        return {
            "success":         False,
            "message":         str(message),
            "carta_detectada": False,
            "aviso_carta":     None,
            "total_tomates":   0,
            "tomates":         [],
            "annotated_image": None,
        }
