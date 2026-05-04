def explain_metrics():
    return {
        "mAP50": "Precision promedio del modelo",
        "Precision": "Predicciones correctas / predicciones totales",
        "Recall": "Objetos detectados correctamente",
        "mAP50-95": "Precision promedio en multiples umbrales IoU (relevante para segmentacion)",
        "mask_mAP50": "mAP50 calculado sobre las mascaras de segmentacion",
    }
