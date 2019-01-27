from math import log

import numpy as np
from PIL import Image
from keras.backend import expand_dims

from keras.engine.saving import load_model, model_from_json

from image_generator import preprocess_image
from word_dictionary import create_dictionaries

dictionary, reversedDictionary = create_dictionaries()
image_size = (224, 224)
log_dir = "logs2/"


def search_beam(model_structure_path, model_weights_path, image, k=3):
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
    sequences = [[list(), 1.0, False]]
    for i in range(1, 50):
        all_candidates = list()
        for j in range(len(sequences)):
            seq, score, is_completed = sequences[j]
            if not is_completed:
                words_input = np.expand_dims(np.append(np.array([st]), seq), axis=0)
                res = model.predict([image, words_input])
                row = res[:, i, :][0]
                for m in range(1, row.size):
                    if row[m] == 0:
                        candidate = [seq + [m], float("inf"), False]
                    else:
                        candidate = [seq + [m], score * -log(row[m]), False]
                    all_candidates.append(candidate)
            else:
                candidate = [seq, 0, is_completed]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # order all candidates by score
        sequences = ordered[:k]  # select k best
        completed = []
        for n in range(len(sequences)):
            if sequences[n][0][len(sequences[n][0])-1] == 1:
                sequences[n][2] = True
            completed.append(sequences[n][2])
        if all(completed):
            break
    return sequences


def get_best_caption(model_structure_path, model_weights_path, image, k=3):
    sequences = search_beam(model_structure_path, model_weights_path, image)
    sequences = sorted(sequences, key=lambda tup: tup[1])
    best_cap= ""
    for i in range(len(sequences)):
        seq = sequences[i]
        predicted_seq = ""
        for j in range(len(seq[0])):
            predicted_seq += (reversedDictionary[seq[0][j]] + " ")
        print(predicted_seq)
        if i == 0:
            best_cap = predicted_seq
    return best_cap


image = Image.open("COCO_test2014_000000003853.jpg")
model_weights_path = log_dir + "nep031-acc0.681-val_acc0.650.h5 "
get_best_caption("model_structure.json", model_weights_path, image)
