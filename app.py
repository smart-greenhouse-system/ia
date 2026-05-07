"""
Aplicación Flask unificada para gestionar tres modelos de IA:
- Lechuga (detección de etapa de crecimiento)
- Moneda (detección y clasificación de monedas)
- Tomate (detección y etapa de crecimiento de tomate cherry)

Cada modelo se ejecuta en un blueprint independiente, permitiendo
mantener la lógica de cada microservicio sin modificaciones.
"""

import sys
from pathlib import Path

from flask import Flask, render_template, jsonify

# Importar los blueprints de cada modelo
from routes.lechuga_routes import lechuga_bp
from routes.moneda_routes import moneda_bp
from routes.tomate_routes import tomate_bp


def create_app():
    """
    Factory function que crea y configura la aplicación Flask.
    Registra todos los blueprints (modelos) y maneja errores globales.
    """
    
    app = Flask(__name__, template_folder='templates')
    
    # Registrar blueprints con prefijos de ruta
    app.register_blueprint(lechuga_bp, url_prefix='/lechuga')
    app.register_blueprint(moneda_bp, url_prefix='/moneda')
    app.register_blueprint(tomate_bp, url_prefix='/tomate')
    
    # Ruta raíz: información general de la API
    @app.route("/", methods=["GET"])
    def home():
        """Endpoint raíz que muestra los modelos disponibles."""
        return jsonify({
            "mensaje": "API de Modelos de IA Unificada",
            "version": "1.0",
            "modelos_disponibles": [
                {
                    "nombre": "Lechuga",
                    "endpoint": "/lechuga/predict",
                    "descripcion": "Detecta etapa de crecimiento de lechuga"
                },
                {
                    "nombre": "Moneda",
                    "endpoint": "/moneda/predict",
                    "descripcion": "Detecta y clasifica monedas en imágenes"
                },
                {
                    "nombre": "Tomate",
                    "endpoint": "/tomate/predict",
                    "descripcion": "Detecta tomate cherry y su etapa de crecimiento"
                }
            ]
        }), 200
    
    # Manejador de errores 404
    @app.errorhandler(404)
    def page_not_found(e):
        """Maneja rutas no encontradas."""
        return render_template('404.html'), 404
    
    # Manejador de errores 500
    @app.errorhandler(500)
    def internal_error(e):
        """Maneja errores internos del servidor."""
        return jsonify({
            "success": False,
            "message": "Error interno del servidor"
        }), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    # Ejecutar con debug=True para desarrollo, cambiar a False en producción
    app.run(debug=True, host="0.0.0.0", port=5000)
