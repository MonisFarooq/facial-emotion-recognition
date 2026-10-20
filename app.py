
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import os
import io

app = Flask(__name__)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "facial_emotion_cnn.keras"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "face_detector",
    "face_detection_yunet_2023mar.onnx"
)

# ==========================================
# 2. LOAD MODELS
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Emotion model not found: {MODEL_PATH}"
    )

if not os.path.exists(FACE_MODEL_PATH):
    raise FileNotFoundError(
        f"YuNet model not found: {FACE_MODEL_PATH}"
    )

emotion_model = load_model(MODEL_PATH)

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    0.8,   # Detection confidence threshold
    0.3,   # NMS threshold
    5000   # Maximum detections
)

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ==========================================
# 3. HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# 4. PREDICT EMOTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image received."
        }), 400

    try:
        file = request.files["image"]

        if not file.filename:
            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400

        image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "The image is empty."
            }), 400

        # Read uploaded image
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = np.array(pil_image)
        height, width = image.shape[:2]

        # Reject blank or nearly uniform images
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if gray.std() < 8:
            return jsonify({
                "success": False,
                "error": "No face detected. Please upload a clear face image."
            }), 400

        # YuNet requires the actual image dimensions
        face_detector.setInputSize((width, height))

        # Detect faces
        _, faces = face_detector.detect(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        if faces is None or len(faces) == 0:
            return jsonify({
                "success": False,
                "error": "No face detected. Please upload an image containing a human face."
            }), 400

        # Choose the largest detected face
        face = max(
            faces,
            key=lambda row: row[2] * row[3]
        )

        x, y, w, h = face[:4]

        # Ignore detections with extremely low confidence
        detection_confidence = float(face[14])

        if detection_confidence < 0.8:
            return jsonify({
                "success": False,
                "error": "No clear face detected. Please try another image."
            }), 400

        # Expand crop slightly around the face
        x, y, w, h = map(int, [x, y, w, h])

        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        face_gray = gray[y1:y2, x1:x2]

        if face_gray.size == 0:
            return jsonify({
                "success": False,
                "error": "Could not extract a face from this image."
            }), 400

        # Prepare CNN input
        face_gray = cv2.resize(face_gray, (48, 48))
        face_array = face_gray.astype(np.float32)
        face_array = np.expand_dims(face_array, axis=(0, -1))

        # Predict emotion
        predictions = emotion_model.predict(
            face_array, verbose=0
        )[0]

        if len(predictions) != len(EMOTIONS):
            return jsonify({
                "success": False,
                "error": "Emotion labels do not match model output."
            }), 500

        index = int(np.argmax(predictions))

        return jsonify({
            "success": True,
            "emotion": EMOTIONS[index].capitalize(),
            "confidence": round(
                float(predictions[index]) * 100, 2
            )
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "success": False,
            "error": "Invalid image. Please upload a valid image file."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing the image."
        }), 500


# ==========================================
# 5. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
