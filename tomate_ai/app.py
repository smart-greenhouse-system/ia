import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template
from inference import TomatoInference

app = Flask(__name__)
model = TomatoInference()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({
            "success": False,
            "message": "No se recibió imagen en formato JSON"
        }), 400

    image_base64 = data["image"]

    result = model.predict_base64(image_base64)

    return jsonify(result)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run(debug=True)
