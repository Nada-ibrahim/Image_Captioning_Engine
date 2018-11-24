"""
Create image generator that feeds coco-dataset images to CNN
Implemented by : Aya Ayman
"""
import json
from collections import defaultdict
import numpy as np
from PIL import Image

def get_random_data(annotation_line, input_shape, proc_img=True):
    '''preprocessing data'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape

    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    if proc_img:
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255.
    return image_data

def image_generator(batch_size, input_shape):
    """
    Generate image batches to feed to CNN
    :param batch_size: size of the batch
    :param input_shape: size of the image
    :return batches of shape (batch_size, input_shape)
    """
    # TODO: image generator by Aya Ayman
    name_box_id = []
    f = open(
        "annotations/captions_val2014.json",
        encoding='utf-8')
    data = json.load(f)
    print("file is opened")
    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = 'val2014/COCO_val2014_%012d.jpg' % id
        name_box_id.append(name)

    f = open('train.txt', 'w')
    for key in name_box_id:
        f.write(key)
        f.write('\n')
    f.close()

    '''data generator for fit_generator'''
    annotation_path='train.txt'
    with open(annotation_path) as f:
        annotation_lines = f.readlines()

    n = len(annotation_lines)
    i=0
    while True:
        image_data = []
        for b in range(batch_size):
            if i==0:
              np.random.shuffle(annotation_lines)
            image= get_random_data(annotation_lines[i], input_shape, proc_img=True)
            image_data.append(image)
            i = (i + 1) % n
            image_data = np.array(image_data)
        yield image_data


