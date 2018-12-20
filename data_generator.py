"""
Creates a data generator that inputs the encoded words and features to the LSTM model
Implemented by: Nada Ibrahim
"""

import numpy as np
from keras.utils import to_categorical
from image_generator import image_generator


def generate(batch_size, dictionary_size, num_steps, image_size, images_path, annotations_path):
    """
    Generate data batches of words and features to feed to LSTM
    :param batch_size: size of batch
    :param dictionary_size: total size of vocabulary
    :param num_steps: number of words in a caption/ number of units
    :param image_size: size of image input to CNN
    :param images_path: paths of the images text file
    :param annotations_path: paths of the encoded captions text file

    """

    # load captions and features from files
    with open(annotations_path) as f:
        annotations = f.readlines()

    # initialize outputs


    img_gen = image_generator(batch_size, image_size, images_path)
    current_idx = 0
    while True:
        image_data = next(img_gen)
        max_len = 0
        x_word = np.zeros((batch_size, num_steps))
        y = np.zeros((batch_size, num_steps + 1, dictionary_size))
        for i in range(batch_size):
            if current_idx >= len(annotations):
                # reset the index back to the start of the data set
                current_idx = 0
            ## process inputs
            caption_words = [int(n) for n in annotations[current_idx].split()]
            if len(caption_words) > max_len:
                max_len =len(caption_words)
            x_word[i, :len(caption_words)] = caption_words
            # pad with <end> tag
            x_word[i, len(caption_words):] = x_word[i, len(caption_words) - 1]

            ## process outputs
            temp_y = np.array(np.append(x_word[i, :], x_word[i, x_word.shape[1]-1]))
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=dictionary_size)

            current_idx += 1
        x_word_crop = x_word[:, :max_len]
        y_crop = y[:,:max_len+1,:]
        yield [image_data, x_word_crop], y_crop
