import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle

from PIL import Image as im

from .load import init

global model, graph
model, graph = init()


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__, static_url_path='')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(image):
    image = np.array(image.resize((32, 32), im.ANTIALIAS)).reshape(-1, 32, 32, 3)
    # resize_image = image, dtype = int).reshape(-1, 32, 32, 3)
    return image

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/identifier')
def upload_file():
    return render_template("identifier.html")

@app.route('/classifier')
def upload_file_classifier():
    return render_template("classifier.html")


@app.route('/upload', methods=['POST'])
def uploaded_file():
    if request.method == 'POST':
        classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherd’s Purse', 'Small-flowered Cranesbill', 'Sugar beet'][::-1]
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save("uploads/" + filename)

        image = preprocess(im.open("uploads/" + filename))
        print("Open image")

        with graph.as_default():
            out = model.predict(image)
            predicted = np.argmax(out, axis=1)[0]
            print(out, predicted)
            result = classes[predicted]
            return str(result)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)


@app.route('/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory('static/fonts', path)


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('static/img', path)
