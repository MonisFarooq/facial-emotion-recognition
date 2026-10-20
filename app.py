import os

# Set resource limits before importing TensorFlow
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

cv2.setNumThreads(1)

# ==========================================
# 1. MODEL PATHS
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
# 2. LOAD MODELS ONCE
# ==========================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Emotion model missing: {MODEL_PATH}")

if not os.path.isfile(FACE_MODEL_PATH):
    raise FileNotFoundError(f"YuNet model missing: {FACE_MODEL_PATH}")

app.logger.info("Loading emotion CNN...")
emotion_model = load_model(MODEL_PATH, compile=False)
app.logger.info("Emotion CNN loaded.")

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# IMPORTANT: Match this list to your training class order.
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
        uploaded_file = request.files["image"]

        if not uploaded_file.filename:
            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400

        image_bytes = uploaded_file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "The uploaded image is empty."
            }), 400

        # Read image and correct orientation
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            pil_image.thumbnail((1280, 1280))
            image = np.asarray(pil_image, dtype=np.uint8)

        height, width = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if gray.std() < 8:
            return jsonify({
                "success": False,
                "error": "Please upload a clear face image."
            }), 400

        # Detect face with YuNet
        face_detector.setInputSize((width, height))
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        _, faces = face_detector.detect(bgr_image)

        if faces is None or len(faces) == 0:
            return jsonify({
                "success": False,
                "error": "No face detected. Upload a clear face image."
            }), 400

        # Select largest detected face
        face = max(faces, key=lambda row: row[2] * row[3])

        x, y, w, h = map(int, face[:4])
        detection_confidence = float(face[14])

        if detection_confidence < 0.8:
            return jsonify({
                "success": False,
                "error": "No clear face detected. Try another image."
            }), 400

        # Add padding around face
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)

        x1 = max(0, x - pad_x)
        y