import math
from pathlib import Path

from keras.backend import expand_dims
# from keras.engine.saving import model_from_json
from keras.models import model_from_json
from im2txt_model import create_training_model
from word_dictionary import create_dictionaries, encode_annotations
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, LearningRateScheduler
from data_generator import generate
from keras.optimizers import adam, SGD

epochs_no = 50


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print("lr => " + str(lrate))
    return lrate


def create_save_model(model_path):
    model = create_training_model(dictionary_size, hidden_size)

    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    return model


def get_model(model_path, weights_path, trainable = False):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={"expand_dims": expand_dims})
    model.load_weights(weights_path)
    if trainable:
        for i in range(1, len(model.layers[1].layers)):
            model.layers[1].layers[i].trainable = True
        model.layers[1].trainable = True
    learning_rate = 0.1
    decay = learning_rate / epochs_no
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, decay=decay, momentum=0.8),
                  metrics=['categorical_accuracy'])
    return model


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

log_dir = "logs4/"
checkpoint = ModelCheckpoint(
    log_dir + 'nep{epoch:03d}-acc{categorical_accuracy:.3f}-val_acc{val_categorical_accuracy:.3f}.h5',
    monitor='val_loss', mode='min', save_weights_only=True, save_best_only=True, period=1)
logging = TensorBoard(log_dir=log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
# lrate = LearningRateScheduler(step_decay)

model_path = Path("model_structure.json")
if model_path.exists():
    model = get_model(model_path, "logs2/nep031-acc0.681-val_acc0.650.h5", trainable=True)
else:
    model = create_save_model(model_path)


model.fit_generator(generate(batch_size, dictionary_size, max_seq_length, image_size, train_path, encoded_train_path)
                    , steps_per_epoch=max(1, num_train // batch_size)
                    , validation_data=generate(batch_size, dictionary_size, max_seq_length, image_size, val_path,
                                               encoded_val_path)
                    , epochs=epochs_no
                    , initial_epoch=0
                    , callbacks=[logging, checkpoint, early_stopping]
                    , validation_steps=max(1, num_val // batch_size))

model.save(log_dir + 'trained_model.h5')

