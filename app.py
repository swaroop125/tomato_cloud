from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model once at startup
# The version downgrade in requirements.txt will fix the 'batch_shape' crash
model = tf.keras.models.load_model("tomato_model.h5")

classes = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy",
    "Tomato_Late_blight", "Tomato_Septoria_leaf_spot", "Tomato_Target_Spot",
    "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Leaf_Mold",
    "Tomato_Mosaic_virus", "Tomato_Yellow_Leaf_Curl_Virus"
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
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocessing: RGB conversion, Resize to 128x128, Normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        
        return jsonify({
            "disease": classes[class_index],
            "confidence": float(np.max(prediction)),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Fix for 'No open ports detected': Use Render's environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
