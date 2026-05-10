import sys
from pathlib import Path

from flask import Blueprint, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent / "predictivo_ai"))
from inference import PredictivoInference

predictivo_bp = Blueprint("predictivo", __name__)
modelo_predictivo = PredictivoInference()

CAMPOS_REQUERIDOS = ['cultivo', 'etapa', 'temperature', 'humidity', 'soil_moisture', 'sunlight']


@predictivo_bp.route("/predict", methods=["POST"])
def predict_predictivo():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "No se recibio JSON en la solicitud"
        }), 400

    faltantes = [c for c in CAMPOS_REQUERIDOS if c not in data]
    if faltantes:
        return jsonify({
            "success": False,
            "message": f"Campos requeridos faltantes: {', '.join(faltantes)}"
        }), 400

    try:
        result = modelo_predictivo.predict(
            cultivo=data['cultivo'],
            etapa=data['etapa'],
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            soil_moisture=float(data['soil_moisture']),
            sunlight=float(data['sunlight'])
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error interno: {str(e)}"
        }), 500
