"""
Create CNN model that outputs the features in the last layer before classification
Implemented by: Shrouk Mansour
"""

from keras.engine.saving import model_from_json
from keras_applications.nasnet import NASNetLarge, NASNetMobile

nas_model = None
def create_nas_model():
    """
    Generate CNN model that takes images as input and output the features
    """

    nas_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for i in range(len(nas_model.layers)):
        nas_model.layers[i].trainable = False

    input_shape = (224, 224)

    return nas_model

def load_nas_model():
    global nas_model
    if nas_model is None:
        json_file = open('obj/nasnet_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        nas_model = model_from_json(loaded_model_json)
        return nas_model
    else:
        return nas_model
