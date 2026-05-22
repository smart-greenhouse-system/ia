def explain_metrics():
    return {
        "mAP50": "Precisión promedio del modelo al 50% de IoU",
        "mAP50-95": "Precisión promedio entre 50% y 95% de IoU (más exigente)",
        "Precision": "Predicciones correctas / predicciones totales",
        "Recall": "Objetos detectados correctamente / objetos reales",
        "mask_mAP50": "mAP50 calculado sobre las máscaras de segmentación",
        "area_foliar_cm2": "Área real de la lechuga calculada con la tarjeta de referencia (6.3x8.8 cm)",
        "resolucion_cm2_por_pixel": "Cuántos cm² representa cada píxel según la tarjeta detectada",
    }
