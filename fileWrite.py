import json
from collections import defaultdict
import numpy as np
from PIL import Image

def file_write():
    # write paths of validaation images on file
    name_box_id = []
    f = open(
        "annotations/captions_val2014.json",
        encoding='utf-8')
    data = json.load(f)
    print("validation captions is opened")
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = 'val2014/val2014/COCO_val2014_%012d.jpg' % id
        name_box_id.append(name)

    f = open('paths/validation.txt', 'w')
    for key in name_box_id:
        f.write(key)
        f.write('\n')
    f.close()


     # write paths of training images
    train_name_box_id = []
    file = open(
        "annotations/captions_train2014.json",
        encoding='utf-8')
    data = json.load(file)
    print("train captions is opened")
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = 'train2014/train2014/COCO_train2014_%012d.jpg' % id
        train_name_box_id.append(name)

    file = open('paths/train.txt', 'w')
    for key in train_name_box_id:
        file.write(key)
        file.write('\n')
    file.close()

file_write()