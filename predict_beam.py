import timeit
from math import log

import numpy as np
from PIL import Image
from keras.backend import expand_dims

from keras.engine.saving import load_model, model_from_json

from im2txt_model import create_training_model
from predictor import Predictor
from word_dictionary import create_dictionaries

dictionary, reversedDictionary = create_dictionaries()
image_size = (224, 224)
log_dir = "logs2/"


def get_best_k(all_candidates, k):
    MAX = float("inf")
    best_k = list()
    firstmin = [list(), MAX]
    secmin = [list(), MAX]
    thirdmin = [list(), MAX]
    for i in range(0, len(all_candidates)):
        if all_candidates[i][1] < firstmin[1]:
            thirdmin = secmin;
            secmin = firstmin;
            firstmin = all_candidates[i]
        elif all_candidates[i][1] < secmin[1]:
            thirdmin = secmin;
            secmin = all_candidates[i]
        elif all_candidates[i][1] < thirdmin[1]:
            thirdmin = all_candidates[i]
    best_k.append([firstmin, secmin, thirdmin])
    return best_k[0]


def search_beam(predictor, image, k=3):
    states = predictor.feed_image(image)
    st = dictionary["<start>"]
    sequences = [[[st], 1.0, states]] * 3
    completed_sequences = []
    max_length = 15
    for i in range(1, max_length):
        all_candidates = list()
        words_input = np.array([sequences[j][0][-1] for j in range(len(sequences))]).reshape(3, 1)
        states = [np.array([sequences[j][2][0] for j in range(len(sequences))]),
                  np.array([sequences[j][2][1] for j in range(len(sequences))])]
        batch_res, batch_states = predictor.predict(words_input, states)
        for j in range(len(sequences)):
            seq, score, _ = sequences[j]
            res = batch_res[j]
            states = [batch_states[0][j], batch_states[1][j]]
            row = res[0, :]
            row = list(enumerate(row))
            row.sort(key=lambda x: -x[1])
            row = row[0:k]
            for w, p in row:
                if w != 2:
                    if p >= 1e-4:
                        if w == 1:
                            candidate = [seq, score * -log(p), states]
                            completed_sequences.append(candidate)
                        else:
                            candidate = [seq + [w], score * -log(p), states]
                            all_candidates.append(candidate)
            if i == 1:
                break
        sequences = get_best_k(all_candidates, k)  # select k best
        completed_sequences = get_best_k(completed_sequences, k)  # select k best
    return completed_sequences


def get_best_caption(model, image, k=3):
    sequences = search_beam(model, image)
    sequences = sorted(sequences, key=lambda tup: tup[1])
    seq = sequences[0]
    predicted_seq = ""
    for j in range(1, len(seq[0])):
        predicted_seq += (reversedDictionary[seq[0][j]] + " ")
    best_cap = predicted_seq
    return best_cap


#image = Image.open("1280px-Abbey_Road_Zebra_crossing_2004-01.jpg")
#model_weights_path = log_dir + "nep031-acc0.681-val_acc0.650.h5"
#predictor = Predictor(model_weights_path, beam_size=3)
#get_best_caption(predictor, image)

#t = timeit.Timer(lambda: get_best_caption(predictor, image))
#print(t.timeit(number=1))
