from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__, static_folder="frontend")
model = load_model("mnist_model.keras")

def preprocess_mnist(image):
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0))
    image = Image.alpha_composite(background, image).convert("L")

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    w, h = image.size
    if w > h:
        new_w = 20
        new_h = max(1, int(h * (20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(w * (20.0 / h)))

    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_image = Image.new("L", (28, 28), 0)
    upper_left_x = (28 - new_w) // 2
    upper_left_y = (28 - new_h) // 2
    new_image.paste(image, (upper_left_x, upper_left_y))

    new_image.save("debug_input.png")

    image_array = np.array(new_image).astype("float32") / 255.0
    return image_array.reshape(1, 28, 28, 1)


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    raw_image = Image.open(io.BytesIO(request.data))
    processed_image = preprocess_mnist(raw_image)

    predictions = model.predict(processed_image, verbose=0)[0]

    results = [
        {"digit": int(i), "probability": float(predictions[i])}
        for i in range(10)
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)