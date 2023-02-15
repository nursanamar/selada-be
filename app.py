from flask import Flask, request
import base64

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from util import base64_to_pil
from PIL import Image
from io import BytesIO
import pickle

from flask_cors import CORS


model = load_model("selada.model")
print('Model loaded. Start serving...')

def model_predict(img, model):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    return predictions


def decode_img(file_path):
    img_dim = 64
    img = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    img = tf.image.resize(img, [img_dim, img_dim])
    # img = tf.reshape(img,[img_dim,img_dim])
    return img

app = Flask(__name__)
CORS(app)

@app.route("/")
def test():
    return "halo"


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    path = f"uploads/{image.filename}"
    image.save(path)
    
    img = decode_img(path)
    pred = model_predict(img,model)

    class_name = pickle.loads(open("labels.pickle","rb").read())
    result = class_name[pred.argmax(axis=1)[0]]

    return {
        "prediction": {
            "lable": result
        }
    }