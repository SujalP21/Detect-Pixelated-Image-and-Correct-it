import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from PIL import Image

MODEL_PATH = 'model/pixelation_cnn_model.h5'

custom_objects = {
    'mse': MeanSquaredError()
}

model = load_model(MODEL_PATH, custom_objects=custom_objects)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

IMAGE_PATH = '/home/loopassembly/Documents/Pixelated/input_images/pixelated-cloud.webp'

input_image = preprocess_image(IMAGE_PATH)

predictions = model.predict(input_image)

threshold = 0.5
if predictions[0][0] >= threshold:
    print('Prediction: Pixelated')
else:
    print('Prediction: Non-pixelated')
