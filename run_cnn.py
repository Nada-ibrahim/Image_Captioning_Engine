import pickle

from cnn_model import create_nas_model
from image_generator import image_generator


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_save_features(images_path, features_path, batch_size):
    model, input_shape = create_nas_model()

    with open(images_path) as f:
        annotation_lines = f.readlines()

    with open(features_path, "w") as f:
        print("Extracting features..")
        generator = image_generator(batch_size, input_shape, images_path)
        i = 0
        write_features = ""
        while i < len(annotation_lines):
            features = model.predict_generator(generator, steps=1)
            flat = features.flatten()
            write_features = write_features + ' '.join(map(str, flat)) + "\n"
            if i % 20 == 0:
                print("Iteration" + str(i) + "...")
            if i % 100 == 0:
                f.write(write_features)
                write_features = ""
                print("--------------\nSaving...\n-----------------")
                features_shape = features.shape
            i = i + 1
        return features_shape


batch_size = 1
# (1,7,7,1056)
features_shape = get_save_features("train.txt", "obj/train_features.txt", batch_size)
get_save_features("val.txt", "obj/val_features", batch_size)
