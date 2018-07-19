import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pickle

from PIL import Image as im


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(image):
    image = image.resize((32, 32), im.ANTIALIAS)
    resize_image = np.asarray(image, dtype = int).reshape(-1)
    return resize_image

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/identifier')
def upload_file():
    return render_template("identifier.html")



@app.route('/upload', methods=['POST'])
def uploaded_file():
    if request.method == 'POST':
        classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherd’s Purse', 'Small-flowered Cranesbill', 'Sugar beet']
        
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(app.config['UPLOAD_FOLDER'] + "/" + filename)

        image = im.open("uploads/" + filename)
        preprocessed_image = preprocess(image)

        model = pickle.load(open("logistic_model.sav", "rb"))
        result = classes[model.predict([preprocessed_image])[0]]
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
