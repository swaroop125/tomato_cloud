from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("tomato_model.h5")

# Order is alphabetical based on your train/ folder names (PlantVillage dataset)
# Run training_set.class_indices to verify this matches your training
classes = [
    "Tomato_Bacterial_spot",        # 0
    "Tomato_Early_blight",          # 1
    "Tomato_Healthy",               # 2
    "Tomato_Late_blight",           # 3
    "Tomato_Leaf_Mold",             # 4
    "Tomato_Septoria_leaf_spot",    # 5
    "Tomato_Spider_mites",          # 6
    "Tomato_Target_Spot",           # 7
    "Tomato_Tomato_mosaic_virus",   # 8
    "Tomato_YellowLeaf_Curl_Virus", # 9
]

@app.route("/")
def home():
    return {"status": "Tomato Disease API is online"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return jsonify({
            "disease": classes[class_index],
            "confidence": round(confidence, 4),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
