from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.activations import softmax


# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and label encoder
model = load_model('models/facial_emotion_detection_model.h5')

class_names = [ '😠 Angry', '🤢 Disgust', '😨 Fear', '😄 Happy', '😐 Neutral', '😢 Sad', '😲 Surprise']


# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert to grayscale (your model expects grayscale 48x48)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (48, 48))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=(0, -1))  # Shape: (1, 48, 48, 1)

    # Predict emotion
    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions)
    confidence_score = round(float(predictions[0][predicted_index]) * 100, 2)
    predicted_label = class_names[predicted_index]

    return predicted_label, confidence_score






@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_label, confidence_score = process_image(file_path)
        return render_template('result.html',
                               image_path=file_path,
                               filename=filename,
                               predicted_label=predicted_label,
                               confidence_score=confidence_score)

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)