from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from inference import TomatoInference

# =========================================================
# CONFIGURACIÓN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# =========================================================
# MODELO IA
# =========================================================

tomato_ai = TomatoInference()

# =========================================================
# REFERENCIA CARTA
# =========================================================
"""
Tamaño real carta:
6.3 x 8.8 cm
"""

CARD_WIDTH_CM = 6.3
CARD_HEIGHT_CM = 8.8

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        "templates",
        "index.html"
    )

# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "success": True,

        "message": "Tomato AI funcionando"
    })

# =========================================================
# PREDICT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "message": "No se recibió JSON"

            }), 400

        image_base64 = data.get("image")

        if not image_base64:

            return jsonify({

                "success": False,

                "message": "No se recibió imagen"

            }), 400

        # ==================================================
        # INFERENCIA
        # ==================================================

        result = tomato_ai.predict_base64(
            image_base64
        )

        if not result.get("success"):

            return jsonify(result), 400

        # ==================================================
        # DETECTAR CARTA
        # ==================================================

        cards = result.get(
            "monedas",
            []
        )

        card_detected = False

        px_per_cm_x = None
        px_per_cm_y = None

        if len(cards) > 0:

            card = cards[0]

            width_px = card.get(
                "width_px"
            )

            height_px = card.get(
                "height_px"
            )

            if width_px and height_px:

                px_per_cm_x = (
                    width_px
                    / CARD_WIDTH_CM
                )

                px_per_cm_y = (
                    height_px
                    / CARD_HEIGHT_CM
                )

                card_detected = True

        # ==================================================
        # TOMATES
        # ==================================================

        processed_tomatoes = []

        for tomato in result.get(
            "tomates",
            []
        ):

            medidas = tomato.get(
                "medidas",
                {}
            )

            area_px = medidas.get(
                "area_px"
            )

            width_px = medidas.get(
                "width_px"
            )

            height_px = medidas.get(
                "height_px"
            )

            diameter_px = medidas.get(
                "diametro_px"
            )

            real_data = {

                "area_cm2": None,

                "ancho_cm": None,

                "alto_cm": None,

                "diametro_cm": None
            }

            # ==============================================
            # CONVERSIÓN REAL
            # ==============================================

            if (
                card_detected
                and px_per_cm_x
                and px_per_cm_y
            ):

                real_width = round(
                    width_px / px_per_cm_x,
                    2
                )

                real_height = round(
                    height_px / px_per_cm_y,
                    2
                )

                avg_px_cm = (
                    px_per_cm_x
                    + px_per_cm_y
                ) / 2

                real_diameter = round(
                    diameter_px
                    / avg_px_cm,
                    2
                )

                area_cm2 = round(
                    area_px
                    / (
                        avg_px_cm ** 2
                    ),
                    2
                )

                real_data = {

                    "area_cm2":
                        area_cm2,

                    "ancho_cm":
                        real_width,

                    "alto_cm":
                        real_height,

                    "diametro_cm":
                        real_diameter
                }

            tomato["medidas_reales"] = (
                real_data
            )

            processed_tomatoes.append(
                tomato
            )

        # ==================================================
        # RESPUESTA
        # ==================================================

        return jsonify({

            "success": True,

            "total_tomates":
                len(processed_tomatoes),

            "clasificacion":
                result.get(
                    "clasificacion"
                ),

            "etapas_detectadas":
                result.get(
                    "etapas_detectadas"
                ),

            "carta_detectada":
                card_detected,

            "referencia_carta_cm": {

                "ancho": CARD_WIDTH_CM,

                "alto": CARD_HEIGHT_CM
            },

            "tomates":
                processed_tomatoes,

            "annotated_image":
                result.get(
                    "annotated_image"
                )
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )