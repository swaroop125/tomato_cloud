from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load model once at startup
# Ensure tomato_model.h5 is in the same directory
model = tf.keras.models.load_model("tomato_model.h5")

# Map these strictly to your training indices
classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    # Add the rest of your 10 classes here
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
        # Read image and convert to RGB
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # CRITICAL: Fix color channels
        
        # Resize to match your model's input (128x128 based on your code)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Inference
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
    # Port 10000 is the default for Render
    app.run(host="0.0.0.0", port=10000)