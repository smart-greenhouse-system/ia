"""
Blueprint para el modelo de Tomate.
Mantiene la lógica original de tomate_ai/app.py sin modificaciones.

Rutas:
- POST /tomate/predict  -> Predicción basada en imagen Base64
"""

import sys
from pathlib import Path

from flask import Blueprint, request, jsonify, render_template

# Importar directamente desde la carpeta del modelo
sys.path.insert(0, str(Path(__file__).parent.parent / "tomate_ai"))
from inference import TomatoInference


# Crear blueprint con nombre único
tomate_bp = Blueprint("tomate", __name__)

# Inicializar modelo al cargar el blueprint
modelo_tomate = TomatoInference()


@tomate_bp.route("/predict", methods=["POST"])
def predict_tomate():
    """
    Endpoint de predicción para tomate cherry.
    Espera JSON con campo 'image' en formato Base64.
    Retorna etapa de crecimiento y días para cosecha.
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
    result = modelo_tomate.predict_base64(image_base64)
    
    return jsonify(result)


@tomate_bp.errorhandler(404)
def tomate_not_found(e):
    """Maneja rutas no encontradas en el blueprint de tomate."""
    return render_template('404.html'), 404
