"""
Creates a data generator that inputs the encoded words and features to the LSTM model
Implemented by: Nada Ibrahim
"""


def word_generator(batch_size, dictionary_size, num_steps):
    """
    Generate data batches of words and features to feed to CNN
    :param batch_size: size of batch
    :param dictionary_size: total size of vocabulary
    :param num_steps: number of words in a caption/ number of units
    """
    # TODO: word_generator by Nada Ibrahim
