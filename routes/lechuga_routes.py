"""
Blueprint para el modelo de Lechuga.
Mantiene la lógica original de lechuga_ai/app.py sin modificaciones.

Rutas:

- POST /lechuga/predict  -> Predicción basada en imagen Base64
"""

import sys
from pathlib import Path

from flask import Blueprint, request, jsonify

# Importar directamente desde la carpeta del modelo
sys.path.insert(0, str(Path(__file__).parent.parent / "lechuga_ai"))
from lechuga_ai.inference import LechugaInference


# Crear blueprint con nombre único
lechuga_bp = Blueprint("lechuga", __name__)

# Inicializar modelo al cargar el blueprint
modelo_lechuga = LechugaInference()


@lechuga_bp.route("/predict", methods=["POST"])
def predict_lechuga():
    """
    Endpoint de predicción para lechuga.
    Espera JSON con campo 'image' en formato Base64.
    Retorna etapa de crecimiento y recomendación.
    """
    
    # Obtener datos JSON enviados por el cliente
    data = request.get_json(silent=True)
    
    # Validar que se recibió la imagen
    if not data or "image" not in data:
        return jsonify({
            "success": False,
            "message": "No se recibió imagen en formato JSON"
        }), 400
    
    image_base64 = data["image"]
    
    # Ejecutar predicción usando la lógica original
    result = modelo_lechuga.predict_base64(image_base64)
    
    return jsonify(result)
