import base64
import io

import cv2
import numpy as np
from PIL import Image


def decode_base64_to_path(image_base64, temp_path):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    with open(temp_path, "wb") as f:
        f.write(image_bytes)


def encode_results_to_base64(results):
    imagen_bgr = results.plot()
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagen_rgb)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=85)
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def calcular_area_foliar(mascaras, clases, nombres, id_tarjeta=3, area_tarjeta_cm2=55.44):
    pixeles_tarjeta = 0
    pixeles_lechuga = 0
    etapa = "No detectada"

    for idx, clase_id in enumerate(clases):
        clase_id_int = int(clase_id)
        nombre = nombres[clase_id_int].lower()

        if clase_id_int == id_tarjeta or "card" in nombre or "tarjeta" in nombre:
            pixeles_tarjeta += int(np.sum(mascaras[idx] > 0))
        else:
            pixeles_lechuga += int(np.sum(mascaras[idx] > 0))
            etapa = nombres[clase_id_int]

    if pixeles_tarjeta == 0 or pixeles_lechuga == 0:
        return None, None, etapa

    factor = area_tarjeta_cm2 / pixeles_tarjeta
    area_cm2 = round(pixeles_lechuga * factor, 2)
    return area_cm2, round(factor, 6), etapa
