"""
Aplicación Flask centralizada - Puerta de entrada única.

Arquitectura:
- Endpoint único: POST /predict
- Recibe imagen + datos de sensores
- Servicio de identificación de cultivos (tomate → lechuga)
- Gestor de sensores almacena y muestra datos

Estructura:
services/
  ├── crop_identifier.py (identifica cultivo)
  └── sensor_manager.py  (gestiona sensores)
"""

import sys
from pathlib import Path

from flask import Flask, request, jsonify

# Importar servicios
from services.crop_identifier import CropIdentifier
from services.sensor_manager import SensorManager


def create_app():
    """
    Factory function que crea y configura la aplicación Flask.
    Inicializa servicios y define endpoint único.
    """
    
    app = Flask(__name__)
    
    # Inicializar servicios
    crop_id = CropIdentifier()
    sensor_mgr = SensorManager()
    
    # ======================================================================
    # ENDPOINT PRINCIPAL ÚNICO - Puerta de entrada
    # ======================================================================
    
    @app.route("/", methods=["GET"])
    def home():
        """
        Endpoint raíz - Información sobre la API.
        """
        return jsonify({
            "mensaje": "API de Análisis de Cultivos - Endpoint Único",
            "version": "2.0",
            "arquitectura": "Análisis centralizado con identificación automática de cultivo",
            "endpoint_principal": "/predict",
            "campos_requeridos": [
                "image (base64)",
                "temperatura (float)",
                "humedad_relativa (float)",
                "humedad_suelo (float)",
                "iluminacion (float)",
                "timestamp (ISO 8601)"
            ]
        }), 200
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Endpoint único de predicción.
        
        Recibe:
            JSON con image, temperatura, humedad_relativa, humedad_suelo, iluminacion, timestamp
        
        Procesa:
            1. Identifica cultivo (tomate o lechuga)
            2. Almacena datos de sensores
            3. Retorna resultado en JSON
        """
        
        # Obtener JSON de la solicitud
        data = request.get_json(silent=True)
        
        # Validar que se recibió JSON
        if not data:
            return jsonify({
                "success": False,
                "error": "No se recibió JSON en la solicitud"
            }), 400
        
        # Validar campos requeridos
        campos_requeridos = ["image", "temperatura", "humedad_relativa", 
                           "humedad_suelo", "iluminacion", "timestamp"]
        faltantes = [c for c in campos_requeridos if c not in data]
        
        if faltantes:
            return jsonify({
                "success": False,
                "error": f"Campos faltantes: {', '.join(faltantes)}"
            }), 400
        
        try:
            # Extraer imagen
            image_base64 = data.get("image")
            
            # Extraer datos de sensores
            temperatura = float(data.get("temperatura"))
            humedad_relativa = float(data.get("humedad_relativa"))
            humedad_suelo = float(data.get("humedad_suelo"))
            iluminacion = float(data.get("iluminacion"))
            timestamp = data.get("timestamp")
            
            # PASO 1: Guardar datos de sensores y mostrar en consola
            sensor_mgr.store_sensor_data(
                temperatura, humedad_relativa, humedad_suelo, 
                iluminacion, timestamp
            )
            
            # PASO 2: Identificar cultivo
            crop_result = crop_id.identify(image_base64)
            
            # PASO 3: Preparar respuesta
            respuesta = {
                "success": crop_result.get("success"),
                "cultivo": crop_result.get("cultivo"),
                "sensores": sensor_mgr.get_sensor_summary()
            }
            
            # Si la identificación fue exitosa, agregar detalles
            if crop_result.get("success"):
                respuesta["detalles_cultivo"] = crop_result.get("detalles")
                return jsonify(respuesta), 200
            else:
                respuesta["error"] = crop_result.get("error")
                respuesta["diagnostico"] = {
                    "tomate": crop_result.get("detalle_tomate"),
                    "lechuga": crop_result.get("detalle_lechuga")
                }
                return jsonify(respuesta), 400
        
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": f"Error en conversión de datos: {str(e)}"
            }), 400
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error interno: {str(e)}"
            }), 500
    
    
    # ======================================================================
    # Manejadores de errores
    # ======================================================================
    # Manejador de errores 404
    @app.errorhandler(404)
    def page_not_found(e):
        """Maneja rutas no encontradas."""
        return jsonify({
            "success": False,
            "error": "Ruta no encontrada. Usa POST /predict"
        }), 404
    
    # Manejador de errores 500
    @app.errorhandler(500)
    def internal_error(e):
        """Maneja errores internos del servidor."""
        return jsonify({
            "success": False,
            "error": "Error interno del servidor"
        }), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    # Ejecutar con debug=True para desarrollo
    app.run(debug=True, host="0.0.0.0", port=5000)
