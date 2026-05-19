import sys
from pathlib import Path

from flask import Blueprint, request, jsonify, render_template

sys.path.insert(0, str(Path(__file__).parent.parent / "riego_ai"))
from inference import VaciadoInference

riego_bp = Blueprint("riego", __name__,
                     template_folder=str(Path(__file__).parent.parent / "riego_ai" / "templates"))
modelo_vaciado = VaciadoInference()


@riego_bp.route("/", methods=["GET"])
def index_riego():
    return render_template('riego.html')


@riego_bp.route("/predict", methods=["POST"])
def predict_riego():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "message": "No se recibio JSON en la solicitud"
        }), 400

    if data.get("desde_bd"):
        result = modelo_vaciado.predict_desde_bd()
        code = 500 if not result.get("success") else 200
        return jsonify(result), code

    faltantes = [c for c in ['cantidad', 'threshold_minimo'] if c not in data]
    if faltantes:
        return jsonify({
            "success": False,
            "message": f"Campos requeridos faltantes: {', '.join(faltantes)}"
        }), 400

    try:
        result = modelo_vaciado.predict(
            cantidad=float(data['cantidad']),
            threshold_minimo=float(data['threshold_minimo']),
            caudal_bomba_lps=float(data.get('caudal_bomba_lps', 0.075))
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
