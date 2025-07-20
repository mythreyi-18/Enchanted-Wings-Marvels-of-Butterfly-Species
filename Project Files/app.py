import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("butterfly_model.h5")

# Load training_set.csv for label mapping
df = pd.read_csv("dataset/training_set.csv")
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
label_map = dict(zip(df['label_encoded'], df['label']))

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in request'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = file.filename
        file_path = os.path.join('static', filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_species = label_map.get(predicted_index, "Unknown")

        return render_template('result.html',
                               prediction=predicted_species,
                               uploaded_image=filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
