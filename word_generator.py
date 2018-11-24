"""
Creates a data generator that inputs the encoded words and features to the LSTM model
Implemented by: Nada Ibrahim
"""

import numpy as np
from keras.utils import to_categorical


def word_generator(batch_size, dictionary_size, num_steps, features_length, features_path, annotations_path):
    """
    Generate data batches of words and features to feed to LSTM
    :param batch_size: size of batch
    :param dictionary_size: total size of vocabulary
    :param num_steps: number of words in a caption/ number of units
    :param features_length: length of the features layer output from CNN
    :param features_path: path of the features text file
    :param annotations_path: path of the encoded captions text file

    """

    # load captions and features from files
    with open(features_path) as f:
        features = f.readlines()
    with open(annotations_path) as f:
        annotations = f.readlines()

    # initialize outputs
    x_word = np.zeros((batch_size, num_steps - 1))
    x_feature = np.zeros((batch_size, features_length))
    y = np.zeros((batch_size, num_steps, dictionary_size))

    current_idx = 0
    while True:
        for i in range(batch_size):
            if current_idx >= len(annotations):
                # reset the index back to the start of the data set
                current_idx = 0

            ## process inputs
            x_feature[i, :] = [float(n) for n in features[current_idx].split()]
            caption_words = [int(n) for n in annotations[current_idx].split()]
            x_word[i, :len(caption_words)] = caption_words
            # pad with <end> tag
            x_word[i, len(caption_words):] = x_word[i, len(caption_words) - 1]

            ## process outputs
            temp_y = np.array(np.append(x_word[i, :], x_word[i, x_word.shape[1]] - 1))
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=dictionary_size)

            current_idx += 1

        yield [x_feature, x_word], y
