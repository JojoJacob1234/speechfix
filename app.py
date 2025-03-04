import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained CNN-LSTM model
MODEL_PATH = "./speech.keras"
  # Change this if your model path is different
model = tf.keras.models.load_model(MODEL_PATH)

# Function to extract MFCC, pitch, RMS, and speaking rate
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

# API Route for Emotion & Confidence Prediction
# ...existing code...

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    print(f"Received file: {file.filename}")  # Debugging print

    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Extract features
    features = extract_features(filepath)
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Predict using the CNN-LSTM model
    predictions = model.predict(features)[0]

    # Define emotion labels (must match the model training labels)
    emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]

    # Convert predictions to percentage probabilities
    emotion_probs = {label: round(float(prob) * 100, 2) for label, prob in zip(emotion_labels, predictions[:-1])}

    # 🔥 Normalize emotion probabilities to sum to 100%
    total = sum(emotion_probs.values())
    if total > 0:
        emotion_probs = {k: round((v / total) * 100, 2) for k, v in emotion_probs.items()}

    # Convert confidence score to High/Medium/Low
    confidence_score = round(float(predictions[-1]) * 100, 2)

    if confidence_score <= 40:
        confidence_level = "Low"
    elif confidence_score <= 70:
        confidence_level = "Medium"
    else:
        confidence_level = "High"

    return jsonify({
        "emotions": emotion_probs,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level
    })


# ...existing code...
# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
    print("Server is running on http://localhost:5000/")