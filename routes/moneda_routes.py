"""
Blueprint para el modelo de Moneda.
Mantiene la lógica original de moneda_ai/app.py sin modificaciones.

Rutas:

- POST /moneda/predict  -> Predicción basada en imagen Base64
"""

import sys
from pathlib import Path

from flask import Blueprint, request, jsonify, render_template

# Importar directamente desde la carpeta del modelo
sys.path.insert(0, str(Path(__file__).parent.parent / "moneda_ai"))
from inference import MonedaInference


# Crear blueprint con nombre único
moneda_bp = Blueprint("moneda", __name__)

# Inicializar modelo al cargar el blueprint
modelo_moneda = MonedaInference()


@moneda_bp.route("/predict", methods=["POST"])
def predict_moneda():
    """
    Endpoint de predicción para moneda.
    Espera JSON con campo 'image' en formato Base64.
    Retorna lista de monedas detectadas y sus clases.
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
    result = modelo_moneda.predict_base64(image_base64)
    
    return jsonify(result)


@moneda_bp.errorhandler(404)
def moneda_not_found(e):
    """Maneja rutas no encontradas en el blueprint de moneda."""
    return render_template('404.html'), 404
