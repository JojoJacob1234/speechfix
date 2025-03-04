import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# 🔹 Force TensorFlow to use only CPU (Prevents CUDA-related errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔹 Initialize Flask app
app = Flask(__name__)
CORS(app)

# 🔹 Define Upload Folder (stores audio files temporarily)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 Load trained CNN-LSTM model
MODEL_PATH = "./speech.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# 🔹 Allowed Audio File Extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

# 🔹 Function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# 🔹 Function to extract MFCC, pitch, RMS, and speaking rate
def extract_features(wav_file_name):
    y, sr = librosa.load(wav_file_name, sr=16000)

    # Extract MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Extract RMS energy (loudness)
    rms = np.mean(librosa.feature.rms(y=y))

    # Extract pitch
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0

    # Compute speaking rate (approximate syllables per second)
    speaking_rate = len(librosa.effects.split(y, top_db=20)) / (len(y) / sr)

    # Combine all extracted features
    features = np.concatenate(([rms, pitch, speaking_rate], mfccs))

    return features

# 🔹 API Route for Emotion & Confidence Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed types: wav, mp3, ogg, flac, m4a"}), 400

    # 🔹 Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # 🔹 Extract features from the audio file
        features = extract_features(filepath)
        features = np.expand_dims(features, axis=0)  # Add batch dimension

        # 🔹 Predict using the CNN-LSTM model
        predictions = model.predict(features)[0]

        # Define emotion labels (must match the model training labels)
        emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]

        # Convert predictions to percentage probabilities
        emotion_probs = {label: round(float(prob) * 100, 2) for label, prob in zip(emotion_labels, predictions[:-1])}

        # 🔹 Normalize emotion probabilities to sum to 100%
        total = sum(emotion_probs.values())
        if total > 0:
            emotion_probs = {k: round((v / total) * 100, 2) for k, v in emotion_probs.items()}

        # 🔹 Convert confidence score to High/Medium/Low
        confidence_score = round(float(predictions[-1]) * 100, 2)

        if confidence_score <= 40:
            confidence_level = "Low"
        elif confidence_score <= 70:
            confidence_level = "Medium"
        else:
            confidence_level = "High"

        # 🔹 Remove file after processing
        os.remove(filepath)

        return jsonify({
            "emotions": emotion_probs,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process the file. {str(e)}"}), 500

# 🔹 Run Flask app (Fix for Render Deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use PORT from environment variable
    app.run(host="0.0.0.0", port=port, debug=True)
