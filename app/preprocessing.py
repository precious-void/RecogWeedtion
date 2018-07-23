from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image as im
import cv2

# Check if it`s photo extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert to HSV and resize to shape (32, 32)
def preprocess(image, scale = 32):
    image_resize = np.asarray(cv2.resize(cv2.imread(image), (scale, scale)))
    image = np.asarray(image_resize, dtype = float).reshape(-1, scale, scale, 3)
    return image

# Segmentation, reshaping
def preprocess_main(image, scale = 70):
    # Reading and resizing
    image = np.asarray(cv2.resize(cv2.imread(image), (scale, scale)))

    # Segmentation
    # Bluring, masking
    blurr = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)
    lower = (25, 40, 50)
    upper = (75, 255, 255)

    mask = cv2.inRange(hsv, lower, upper)
    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc)
    boolean = mask > 0
    new = np.zeros_like(image, np.uint8)
    new[boolean] = image[boolean]

    return (new / 255).reshape(-1, scale, scale, 3)

# Save and check whether it in allowed extensions or not
def save_file(file):
    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return False

    file.save("uploads/" + filename)
    return filename