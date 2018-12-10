import timeit

import numpy as np

from PIL import Image

from im2txt_model import create_model
from image_generator import preprocess_image
from word_dictionary import create_dictionaries, encode_annotations
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from data_generator import generate

FilePath1 = "annotations/captions_train2014.json"
FilePath2 = "annotations/captions_val2014.json"

dictionary, reversedDictionary = create_dictionaries()
encode_annotations(FilePath1, 'annotations/encoded_train_annotations.txt', dictionary)
encode_annotations(FilePath2, 'annotations/encoded_val_annotations.txt', dictionary)

dictionary_size = len(dictionary)
print(dictionary_size)
max_seq_length = 52
hidden_size = 512
batch_size = 10
image_size = (224, 224)
train_path = "paths/train.txt"
val_path = "paths/validation.txt"

encoded_train_path = "annotations/encoded_train_annotations.txt"
encoded_val_path = "annotations/encoded_val_annotations.txt"

with open(train_path) as f:
    annotation_lines = f.readlines()
num_train = len(annotation_lines)

with open(val_path) as f:
    annotation_lines = f.readlines()
num_val = len(annotation_lines)
del annotation_lines

log_dir = "logs/"
# logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{categorical_accuracy:.3f}-val_acc{val_categorical_accuracy:.3f}.h5',
        monitor='val_categorical_accuracy', mode='max', save_weights_only=True, save_best_only=True, period=1)
# checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#                              monitor='val_acc', save_best_only=True, save_weights_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=10, verbose=1)

model = create_model(dictionary_size, max_seq_length, hidden_size)

model.fit_generator(generate(batch_size, dictionary_size, max_seq_length, image_size, train_path, encoded_train_path)
                    , steps_per_epoch=max(1, num_train // batch_size)
                    , validation_data=generate(batch_size, dictionary_size, max_seq_length, image_size, val_path,
                                               encoded_val_path)
                    , epochs=100
                    , initial_epoch=0
                    , callbacks=[checkpoint, reduce_lr, early_stopping]
                    , validation_steps=max(1, num_val // batch_size))

model.save(log_dir + 'trained_model.h5')

# stateful_model = create_model(dictionary_size, max_seq_length, hidden_size, stateful=True)
# stateful_model.load_weights(log_dir + 'trained_model.h5')
# #inference
# print("Inference test")
#
# def infer(image):
#
#     image = np.expand_dims(preprocess_image(image, image_size, proc_img=True),axis=0)
#     st = dictionary["<start>"]
#     words = np.array([st])
#     words_input = np.expand_dims(words, axis=0)
#     # words = np.append(words, st)
#     for i in range(1,15):
#         res = stateful_model.predict([image, words_input])
#         t = timeit.Timer(lambda: model.predict([image, words_input]))
#         print(t.timeit(number=1))
#         arg = np.argmax(res[:, 1, :])
#         predicted_word = reversedDictionary[arg]
#         print(str(i) + "- " + predicted_word)
#         if predicted_word == "<end>":
#             break
#         words = np.array([arg])
#         words_input = np.expand_dims(words, axis=0)
#
# image = Image.open("image.jpg")
# t = timeit.Timer(lambda: infer(image) )
# print(t.timeit(number=1))
