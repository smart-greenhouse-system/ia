import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from riego_ai.model import VaciadoModel
from riego_ai.utils.preprocessing import (
    validar_entrada,
    calcular_ratio,
    formatear_tiempo
)
from riego_ai.utils.constants import (
    MONGO_URI,
    NOMBRE_TANQUE,
    CAUDAL_BOMBA_DEFECTO
)


class VaciadoInference:

    def __init__(self):
        models_dir = BASE_DIR / "models"
        self.model = VaciadoModel()
        self.model.load(str(models_dir / "modelo_vaciado.pkl"))

    def predict(self, cantidad, threshold_minimo, caudal_bomba_lps):
        validar_entrada(cantidad, threshold_minimo, caudal_bomba_lps)

        volumen = cantidad - threshold_minimo
        ratio = calcular_ratio(cantidad, threshold_minimo, caudal_bomba_lps)
        tiempo = float(self.model.predict([[ratio]])[0])
        tiempo = max(0, tiempo)

        return {
            "success": True,
            "nivel_actual": cantidad,
            "nivel_minimo": threshold_minimo,
            "volumen_a_vaciar": round(volumen, 1),
            "caudal_bomba": caudal_bomba_lps,
            "tiempo_segundos": round(tiempo, 1),
            "tiempo_formateado": formatear_tiempo(tiempo),
            "estado": "NORMAL"
        }

    def predict_desde_bd(self):
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI)
            db = client.get_default_database()
            inventario = db.inventory.find_one({"nombre": NOMBRE_TANQUE})
            client.close()
        except Exception as e:
            return {
                "success": False,
                "message": f"Error al conectar con MongoDB: {str(e)}"
            }

        if not inventario:
            return {
                "success": False,
                "message": f"No se encontro el recurso '{NOMBRE_TANQUE}' en la base de datos"
            }

        cantidad = float(inventario.get('cantidad', 0))
        threshold = float(inventario.get('threshold_minimo', 0))
        caudal = float(inventario.get('caudal_bomba_lps', CAUDAL_BOMBA_DEFECTO))

        if cantidad <= threshold:
            return {
                "success": True,
                "nivel_actual": cantidad,
                "nivel_minimo": threshold,
                "volumen_a_vaciar": 0,
                "caudal_bomba": caudal,
                "tiempo_segundos": 0,
                "tiempo_formateado": "0 segundos",
                "estado": "TANQUE_EN_NIVEL_MINIMO"
            }

        return self.predict(cantidad, threshold, caudal)
