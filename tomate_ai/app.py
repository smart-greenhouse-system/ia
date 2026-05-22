import traceback
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory
)

from flask_cors import CORS

from inference import TomatoInference


# =========================================================
# BASE DIR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

print(f"\n📁 BASE_DIR:\n{BASE_DIR}")


# =========================================================
# FLASK CONFIG
# =========================================================

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)

app.config["JSON_SORT_KEYS"] = False

print("\n🚀 FLASK INICIADO")


# =========================================================
# LOAD AI
# =========================================================

tomato_ai = None

try:

    print("\n🤖 CARGANDO TOMATO AI...")

    tomato_ai = TomatoInference()

    print("✅ IA CARGADA")

except Exception as e:

    print("\n❌ ERROR CARGANDO IA")

    print(e)

    traceback.print_exc()


# =========================================================
# HELPERS
# =========================================================

def json_success(data, status=200):

    return jsonify(data), status


def json_error(message, status=400):

    print(f"\n❌ JSON ERROR: {message}")

    return jsonify({

        "success": False,

        "message": str(message)

    }), status


# =========================================================
# ROOT
# =========================================================

@app.route("/", methods=["GET"])
def home():

    try:

        print("\n🏠 HOME")

        index_path = (
            BASE_DIR /
            "templates" /
            "index.html"
        )

        if not index_path.exists():

            print(
                f"❌ index.html NO EXISTE:\n"
                f"{index_path}"
            )

            return json_error(
                "index.html no encontrado",
                404
            )

        print("✅ FRONTEND OK")

        return send_from_directory(
            app.template_folder,
            "index.html"
        )

    except Exception as e:

        print("\n❌ HOME ERROR")

        print(e)

        traceback.print_exc()

        return json_error(
            "Error cargando frontend",
            500
        )


# =========================================================
# HEALTH
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    try:

        print("\n💚 HEALTH CHECK")

        ai_loaded = (
            tomato_ai is not None
        )

        response = {

            "success":
                True,

            "status":
                "online",

            "ai_loaded":
                ai_loaded,

            "device":
                (
                    str(tomato_ai.DEVICE)
                    if ai_loaded
                    else None
                ),

            "segment_model":
                (
                    str(
                        tomato_ai
                        .SEGMENT_MODEL_PATH
                        .name
                    )
                    if ai_loaded
                    else None
                ),

            "growth_model":
                (
                    str(
                        tomato_ai
                        .GROWTH_MODEL_PATH
                        .name
                    )
                    if ai_loaded
                    else None
                ),

            "homography":
                True,

            "real_measurements":
                True,

            "pixels_per_cm":
                (
                    tomato_ai.PIXELS_PER_CM
                    if ai_loaded
                    else None
                )
        }

        print("✅ HEALTH OK")

        return json_success(response)

    except Exception as e:

        print("\n❌ HEALTH ERROR")

        print(e)

        traceback.print_exc()

        return json_error(
            "Health check failed",
            500
        )


# =========================================================
# API INFO
# =========================================================

@app.route("/api/info", methods=["GET"])
def api_info():

    try:

        print("\n📘 API INFO")

        if tomato_ai is None:

            return json_error(
                "IA no disponible",
                500
            )

        response = {

            "success": True,

            "project":
                "Tomato AI Vision",

            "version":
                "4.0",

            "device":
                tomato_ai.DEVICE,

            "segmentation_model":
                str(
                    tomato_ai
                    .SEGMENT_MODEL_PATH
                ),

            "growth_model":
                str(
                    tomato_ai
                    .GROWTH_MODEL_PATH
                ),

            "segment_classes":
                tomato_ai
                .segment_model
                .names,

            "growth_classes":
                tomato_ai
                .growth_model
                .names,

            "card_reference": {

                "width_cm":
                    tomato_ai
                    .CARD_WIDTH_CM,

                "height_cm":
                    tomato_ai
                    .CARD_HEIGHT_CM
            },

            "homography": {

                "enabled":
                    True,

                "pixels_per_cm":
                    tomato_ai
                    .PIXELS_PER_CM,

                "warp_width":
                    tomato_ai
                    .WARP_WIDTH,

                "warp_height":
                    tomato_ai
                    .WARP_HEIGHT
            },

            "real_measurements": {

                "area":
                    "cm2",

                "diameter":
                    "cm",

                "perimeter":
                    "cm"
            },

            "harvest_days":
                tomato_ai.harvest_map
        }

        print("✅ API INFO OK")

        return json_success(response)

    except Exception as e:

        print("\n❌ API INFO ERROR")

        print(e)

        traceback.print_exc()

        return json_error(
            "Error obteniendo información",
            500
        )


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print(
            "\n"
            "=================================================="
        )

        print("📩 NUEVA REQUEST /predict")

        print(
            "=================================================="
        )

        # =====================================================
        # IA VALIDATION
        # =====================================================

        if tomato_ai is None:

            return json_error(
                "La IA no fue cargada",
                500
            )

        # =====================================================
        # JSON VALIDATION
        # =====================================================

        data = request.get_json(
            force=False,
            silent=True
        )

        if not data:

            print("❌ JSON VACÍO")

            return json_error(
                "JSON inválido"
            )

        image_base64 = data.get("image")

        if not image_base64:

            print("❌ IMAGE VACÍA")

            return json_error(
                "Imagen requerida"
            )

        print("✅ JSON OK")

        print(
            f"📦 SIZE BASE64: "
            f"{len(image_base64)}"
        )

        # =====================================================
        # INFERENCE
        # =====================================================

        print("\n🧠 EJECUTANDO INFERENCIA...")

        result = (
            tomato_ai.predict_base64(
                image_base64
            )
        )

        print("\n📊 RESULTADO IA")

        print(result.keys())

        # =====================================================
        # RESULT VALIDATION
        # =====================================================

        if not isinstance(result, dict):

            return json_error(
                "Resultado inválido",
                500
            )

        if not result.get("success"):

            print("\n❌ ERROR INFERENCIA")

            print(result)

            return json_error(

                result.get(
                    "message",
                    "Error inferencia"
                ),

                500
            )

        # =====================================================
        # DEBUG RESPONSE
        # =====================================================

        total = result.get(
            "total_tomates",
            0
        )

        carta = result.get(
            "carta_detectada",
            False
        )

        pixels_per_cm = result.get(
            "pixels_per_cm"
        )

        print("\n🍅 RESULTADOS")

        print(
            f"🍅 TOTAL TOMATES: "
            f"{total}"
        )

        print(
            f"🪪 CARTA DETECTADA: "
            f"{carta}"
        )

        print(
            f"📏 PIXELS/CM: "
            f"{pixels_per_cm}"
        )

        # =====================================================
        # PER TOMATO DEBUG
        # =====================================================

        tomatoes = result.get(
            "tomates",
            []
        )

        for tomato in tomatoes:

            print(
                "\n----------------------------"
            )

            print(
                f"🍅 TOMATE ID: "
                f"{tomato.get('id')}"
            )

            print(
                f"🌱 ESTADO: "
                f"{tomato.get('estado')}"
            )

            print(
                f"📅 DÍAS COSECHA: "
                f"{tomato.get('dias_cosecha')}"
            )

            medidas = tomato.get(
                "medidas_reales",
                {}
            )

            print(
                f"📐 ÁREA: "
                f"{medidas.get('area_cm2')} cm2"
            )

            print(
                f"📏 DIÁMETRO: "
                f"{medidas.get('diametro_cm')} cm"
            )

            print(
                f"📎 PERÍMETRO: "
                f"{medidas.get('perimetro_cm')} cm"
            )

        print(
            "\n✅ REQUEST COMPLETADA"
        )

        return json_success(result)

    except Exception as e:

        print(
            "\n"
            "=================================================="
        )

        print("❌ PREDICT ERROR")

        print(
            "=================================================="
        )

        print(e)

        traceback.print_exc()

        return json_error(
            str(e),
            500
        )


# =========================================================
# STATIC FILES
# =========================================================

@app.route(
    "/static/<path:filename>",
    methods=["GET"]
)
def static_files(filename):

    try:

        print(
            f"\n📂 STATIC FILE: "
            f"{filename}"
        )

        static_path = (
            BASE_DIR /
            "static" /
            filename
        )

        if not static_path.exists():

            print(
                f"❌ STATIC NO EXISTE:\n"
                f"{static_path}"
            )

            return json_error(
                "Archivo no encontrado",
                404
            )

        return send_from_directory(
            app.static_folder,
            filename
        )

    except Exception as e:

        print("\n❌ STATIC ERROR")

        print(e)

        traceback.print_exc()

        return json_error(
            "Error archivo static",
            500
        )


# =========================================================
# 404
# =========================================================

@app.errorhandler(404)
def not_found(e):

    print("\n❌ 404")

    return json_error(
        "Ruta no encontrada",
        404
    )


# =========================================================
# 413
# =========================================================

@app.errorhandler(413)
def too_large(e):

    print("\n❌ ARCHIVO MUY GRANDE")

    return json_error(
        "Imagen demasiado grande",
        413
    )


# =========================================================
# 500
# =========================================================

@app.errorhandler(500)
def internal_error(e):

    print("\n❌ INTERNAL SERVER ERROR")

    traceback.print_exc()

    return json_error(
        "Error interno servidor",
        500
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "\n"
        "=================================================="
    )

    print("🔥 TOMATO AI VISION v4.0")

    print(
        "=================================================="
    )

    print("🌐 URL:")
    print("http://0.0.0.0:5000")

    print("\n🧠 IA:")
    print("YOLO SEGMENTATION")
    print("YOLO GROWTH CLASSIFICATION")

    print("\n📐 GEOMETRÍA:")
    print("HOMOGRAFÍA ACTIVADA")
    print("MEDIDAS REALES EN CM²")

    print("\n🍅 FEATURES:")
    print("- Área real tomate")
    print("- Diámetro real")
    print("- Perímetro real")
    print("- Corrección perspectiva")
    print("- Días estimados cosecha")
    print("- Segmentación rectificada")

    print("\n🚀 INICIANDO FLASK...")

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True,

        threaded=True,

        use_reloader=False
    )