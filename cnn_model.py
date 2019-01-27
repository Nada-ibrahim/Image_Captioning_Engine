"""
Create CNN model that outputs the features in the last layer before classification
Implemented by: Shrouk Mansour
"""
import keras
from keras_applications.nasnet import NASNetLarge, NASNetMobile, preprocess_input
from keras.utils import plot_model


def create_nas_model():
    """
    Generate CNN model that takes images as input and output the features
    """

    nas_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for i in range(len(nas_model.layers)):
        nas_model.layers[i].trainable = False

    input_shape = (224, 224)
    plot_model(nas_model, 'nas_net_model.png')
    return nas_model

create_nas_model()
