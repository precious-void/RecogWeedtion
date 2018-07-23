import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory

import numpy as np
from PIL import Image as im
import tensorflow as tf
from keras.backend import clear_session

from .load import *
from .preprocessing import *



app = Flask(__name__, static_url_path='')


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/recognizer')
def recognizer():
    return render_template("recognizer.html")

@app.route('/classifier')
def classifier():
    return render_template("classifier.html")


@app.route('/upload_recognizer', methods=['POST'])
def upload_recognizer():
    if request.method == 'POST':
        classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherd’s Purse', 'Small-flowered Cranesbill', 'Sugar beet']
        filename = save_file(request.files['file'])

        if not filename:
            return "False"
        
        # Preprocess image
        image = preprocess_main("uploads/" + filename)

        # Init model and graph
        model, graph = init_recognition_cnn()

        # Predict
        with graph.as_default():
            result = classes[np.argmax(model.predict(image), axis = 1)[0]]
            del model, graph
            return str(result)

@app.route('/upload_classifier', methods=['POST'])
def upload_classifier():
    if request.method == 'POST':
        # Recieved Data {class_number, file}
        recieved_class = int(request.form['number'])
        filename = save_file(request.files['file'])

        # Check if in allowed extentions
        if not filename:
            return "False"

        model_files = ['Black-grass.h5', 'Charlock.h5', 'Cleavers.h5', 'Common_Chickweed.h5', 'Common_wheat.h5', 'Fat_Hen.h5', 'Loose_Silky-bent.h5', 'Maize.h5', 'Scentless_Mayweed.h5', 'Shepherd’s_Purse.h5', 'Small-flowered_Cranesbill.h5', 'Sugar_beet.h5']
        
        # Preprocessing image
        image = preprocess("uploads/" + filename)

        model, graph = init_neural_network("CNN/" + model_files[recieved_class])
        
        # Predict
        with graph.as_default():
            result = np.argmax(model.predict(image), axis = 1)[0]
            del model, graph
            return str(result + 1)
