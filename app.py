# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:56:17 2023

@author: Sanjana

"""

from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np
import re
#reate a Flask application object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static\\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'malaria_model.h5'
model = load_model(MODEL_PATH)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='Error: No image file found.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='Error: No image file selected.')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
            # Perform your prediction here
        img = image.load_img(file_path, target_size=(130, 130))
        img = image.img_to_array(img)
        img = img.reshape((1, 130, 130, 3))

        pred_result= model.predict(img)
        predictions = pred_result > 0.7
        predict1d=predictions.flatten()
        counts = np.bincount(predict1d)
        predicted_class = np.argmax(counts)

        if predicted_class==0:
            prediction='Infected'
        else:
           prediction= 'Not Infected'

        
         # Replace this with your actual prediction
        file_path = re.sub(r'\\', '/', file_path)
      
        return render_template('index.html', prediction=prediction, image_path=file_path)

    else:
        return "Invalid file format. Only PNG, JPG, and JPEG files are allowed."

# Main route to render the form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

