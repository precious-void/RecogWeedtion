from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image as im

# Check if it`s photo extension`
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert to RGB and resize to shape (32, 32)
def preprocess(image):
    image = image.convert('RGB')
    image_resize = image.resize((32, 32), im.ANTIALIAS)
    image = np.asarray(image_resize, dtype = float).reshape(-1, 32, 32, 3)
    return image

# Save and check whether it in allowed extensions or not
def save_file(file):
    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return False

    file.save("uploads/" + filename)
    return filename