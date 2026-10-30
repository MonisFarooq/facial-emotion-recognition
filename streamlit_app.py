
import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EmotionAI | Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
/* Background */
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #201747 0%, transparent 38%),
        radial-gradient(ellipse at 95% 85%, #171b3b 0%, transparent 35%),
        #090c16;
    color: #f5f3ff;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding: 2rem 2rem 3rem;
}

/* Hero section */
.hero {
    padding: 25px 0 30px;
}

.hero-label {
    color: #b4a5ff;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    font-size: clamp(2.5rem, 5vw, 4.2rem);
    line-height: 1.1;
    font-weight: 850;
    letter-spacing: -2px;
    color: #ffffff;
    margin: 16px 0;
}

.accent {
    background: linear-gradient(90deg, #a78bfa, #e879f9, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    color: #aab3ca;
    font-size: 1.05rem;
    line-height: 1.8;
}

/* Headings */
h2, h3 {
    color: #f5f3ff !important;
    font-weight: 750 !important;
}

p, label {
    color: #c8cce0;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #a5aec5;
    font-weight: 650;
    padding: 14px 18px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c4b5fd;
}

div[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    height: 3px;
}

div[data-baseweb="tab-border"] {
    background: #292d44;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #211b40, #151a2b);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 18px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: #aaa7c9;
}

div[data-testid="stMetricValue"] {
    color: #c4b5fd;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(110deg, #7c3aed, #9333ea);
    color: white;
    border: 1px solid #a78bfa;
    border-radius: 12px;
    padding: 9px 18px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #e9d5ff;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    color: white;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(22, 27, 48, 0.75);
    border: 1px dashed #7663bd;
    border-radius: 16px;
    padding: 16px;
}

/* Images and webcam */
[data-testid="stImage"] img {
    border-radius: 16px;
}

video {
    border-radius: 16px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Divider and footer */
hr {
    border-color: #292d44;
    margin: 28px 0;
}

.footer {
    text-align: center;
    color: #858eaa;
    padding: 18px 0;
    font-size: 0.85rem;
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero h1 {
        letter-spacing: -1px;
    }

    button[data-baseweb="tab"] {
        padding: 10px 6px;
        font-size: 0.85rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. MODEL CONFIGURATION
# =====================================================

MODEL_PATH = os.path.join(
    "models",
    "facial_emotion_model.keras"
)

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Could not load the OpenCV face detector."
        )

    return detector


try:
    model = load_model()
    face_detector = load_face_detector()

    if model.output_shape[-1] != len(CLASS_NAMES):
        st.error(
            "The model output does not match the seven emotion labels. "
            "Check the class order used during training."
        )
        st.stop()

except Exception as exc:
    st.error(f"Could not load the model or face detector: {exc}")
    st.stop()


# =====================================================
# 4. EMOTION PREDICTION
# =====================================================

def predict_face(gray_face):
    face = cv2.resize(gray_face, (48, 48))

    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    predictions = model.predict(face, verbose=0)[0]

    index = int(np.argmax(predictions))
    emotion = CLASS_NAMES[index]
    confidence = float(predictions[index]) * 100

    return emotion, confidence


def analyze_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    results = []

    for x, y, w, h in faces:
        face = gray[y:y + h, x:x + w]

        emotion, confidence = predict_face(face)

        results.append({
            "box": (x, y, w, h),
            "emotion": emotion,
            "confidence": confidence
        })

    return results


# =====================================================
# 5. LIVE WEBCAM PROCESSOR
# =====================================================

class EmotionVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_emotion = "Waiting for face"
        self.latest_confidence = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Resize large frames to improve processing speed
        height, width = image.shape[:2]

        if width > 640:
            new_height = int(height * 640 / width)
            image = cv2.resize(image, (640, new_height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) > 0:
            # Process the largest face
            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

            emotion, confidence = predict_face(
                gray[y:y + h, x:x + w]
            )

            with self.lock:
                self.latest_emotion = emotion
                self.latest_confidence = confidence

            # Draw face bounding box
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (190, 140, 255),
                2
            )

            label = f"{emotion.upper()} | {confidence:.1f}%"
            label_y = max(y - 12, 25)

            # Label background
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                image,
                (x, label_y - text_height - 10),
                (x + text_width + 10, label_y + 5),
                (70, 35, 105),
                -1
            )

            cv2.putText(
                image,
                label,
                (x + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        else:
            with self.lock:
                self.latest_emotion = "No face detected"
                self.latest_confidence = 0.0

            cv2.putText(
                image,
                "No face detected",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 190, 255),
                2,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =====================================================
# 6. HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">
    <p class="hero-label">
        AI · COMPUTER VISION · DEEP LEARNING
    </p>
    <h1>
        Understand emotions.<br>
        <span class="accent">In real time.</span>
    </h1>
    <p class="subtitle">
        Facial emotion recognition powered by your trained CNN model.
        Start your camera to see predictions update live.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. DEFINE TABS BEFORE USING THEM
# =====================================================

live_tab, upload_tab = st.tabs([
    "🎥 Live Webcam Detection",
    "🖼️ Image Prediction"
])


# =====================================================
# 8. LIVE WEBCAM TAB
# =====================================================

with live_tab:

    st.subheader("Live emotion detection")

    st.write(
        "Allow camera access and face the camera. "
        "The face box and emotion label update as frames are processed."
    )

    ctx = webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if ctx.state.playing and ctx.video_processor is not None:

        processor = ctx.video_processor

        with processor.lock:
            current_emotion = processor.latest_emotion
            current_confidence = processor.latest_confidence

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Detected emotion",
                current_emotion.title()
            )

        with col2:
            if current_emotion in CLASS_NAMES:
                st.metric(
                    "Model confidence",
                    f"{current_confidence:.1f}%"
                )
            else:
                st.metric(
                    "Model confidence",
                    "--"
                )

    st.caption(
        "Tip: Face the camera in a well-lit environment for better detection."
    )


# =====================================================
# 9. IMAGE PREDICTION TAB
# =====================================================

with upload_tab:

    st.subheader("Analyze a face image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image containing one or more visible faces."
    )

    if uploaded_file is not None:

        file_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error(
                "Could not read this image. Please upload another."
            )

        else:
            results = analyze_image(image)
            display_image = image.copy()

            for result in results:
                x, y, w, h = result["box"]

                label = (
                    f'{result["emotion"].upper()} '
                    f'| {result["confidence"]:.1f}%'
                )

                cv2.rectangle(
                    display_image,
                    (x, y),
                    (x + w, y + h),
                    (190, 140, 255),
                    2
                )

                cv2.putText(
                    display_image,
                    label,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (190, 140, 255),
                    2,
                    cv2.LINE_AA
                )

            col1, col2 = st.columns([1.4, 1])

            with col1:
                st.image(
                    cv2.cvtColor(
                        display_image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Detected faces and predictions",
                    use_container_width=True
                )

            with col2:
                if not results:
                    st.warning(
                        "No face detected. Try a clearer, "
                        "front-facing image."
                    )

                else:
                    for number, result in enumerate(results, start=1):
                        st.markdown(f"### Face {number}")

                        st.metric(
                            "Predicted emotion",
                            result["emotion"].title()
                        )

                        st.metric(
                            "Model confidence",
                            f'{result["confidence"]:.1f}%'
                        )


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    EmotionAI · CNN · TensorFlow · OpenCV · Streamlit
</div>
""", unsafe_allow_html=True)
