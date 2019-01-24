import os
import timeit

import numpy as np
from PIL import Image
from keras.backend import expand_dims

from keras.engine.saving import load_model, model_from_json

from image_generator import preprocess_image
from word_dictionary import create_dictionaries

dictionary, reversedDictionary = create_dictionaries()
image_size = (224, 224)
log_dir = "logs2/"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def infer(model_structure_path, model_weights_path, image):
    json_file = open(model_structure_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={"expand_dims": expand_dims})
    model.load_weights(model_weights_path)

    image = np.expand_dims(preprocess_image(image, image_size, proc_img=True),axis=0)
    st = dictionary["<start>"]
    words = np.array([st])
    words_input = np.expand_dims(words, axis=0)
    # words = np.append(words, st)
    for i in range(1,50):
        res = model.predict([image, words_input])
        t = timeit.Timer(lambda: model.predict([image, words_input]))
        # print(t.timeit(number=1))
        # print(*res[1][0, 2800:3000])
        arg = np.argmax(res[:, i, :])
        predicted_word = reversedDictionary[arg]
        print(predicted_word)
        if predicted_word == "<end>":
            break
        words = np.append(words, arg)
        words_input = np.expand_dims(words, axis=0)


image = Image.open("1280px-Abbey_Road_Zebra_crossing_2004-01.jpg")
model_weights_path = log_dir + "nep031-acc0.681-val_acc0.650.h5 "
infer("model_structure.json", model_weights_path, image)
