"""
Create CNN model that outputs the features in the last layer before classification
Implemented by: Shrouk Mansour
"""

from keras import applications
from keras_applications.nasnet import NASNetLarge, NASNetMobile, preprocess_input
from keras_preprocessing import image
import numpy as np


def generate_cnn_model():
    """
    Generate CNN model that takes images as input and output the features
    """
    # TODO: CNN model by Shrouk Mansour

    model = NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    img = image.load_img('keras.png', target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    features = model.predict(img_data)

    print(features.shape)
    return features

generate_cnn_model()