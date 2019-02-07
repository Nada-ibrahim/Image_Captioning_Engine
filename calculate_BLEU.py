from __future__ import division

import math
import os
from keras.backend import expand_dims
#from get_model import get_model
from predictor import Predictor
from predict_beam import get_best_caption
from PIL import Image
import re
import json
from nltk import word_tokenize
import nltk.compat
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams


def extractCaptions(FilePath):

    txt = " "
    captions=[]
    dictionary = {}
    with open(FilePath,'r') as json_data:
     data = json.load(json_data)
    annotations = data['annotations']
    for record in annotations:
        captions = []
        id = record['image_id']
        name = 'val2014/COCO_val2014_%012d.jpg' % id
        txt = record['caption']
        txt=txt.lower()
        txt.strip()
        #txt=re.split(r'\s',txt)
        txt = re.findall(r"[\w']+", txt)
        if txt == None:
            continue
        if name in dictionary.keys():
            value = dictionary[name]
            if value==None:
                dictionary.pop(name)
                captions.append(txt)
                newItem = {name: captions}
                dictionary.update(newItem)
                continue
            value.append(txt)
            dictionary[name] = value
        else:
            captions.append(txt)
            newItem = {name: captions}
            dictionary.update(newItem)

    return dictionary

def calculate_BLEU(filePath):
    current_idx=0
    weights = [1,0,0,0]
    log_dir = "modified logs2/"
    dictionary=extractCaptions(filePath)
    print(len(dictionary))
    for root, subdirs, files in os.walk(log_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            counter=0
            count=0
            model_weights_path = file_path
            print("------------------------------------------------------")
            print("loading model "+ file_path)
            predictor = Predictor(model_weights_path, beam_size=3)
            print("finished")
            print("------------------------------------------------------")

            for key,value in dictionary.items():
                #print(key)
                path="E:/data set/"
                #print(path+key)
                image = Image.open(path+key)
                #model_weights_path = log_dir + "nep031-acc0.681-val_acc0.650.h5 "
                #model = get_model("model_structure.json", model_weights_path, image)

                candidate=get_best_caption(predictor,image)
                #print(candidate)
                candidate_words=candidate.split(" ")
                #print(candidate_words)
                refrence=value
                #print("ref",refrence)
                score = sentence_bleu(refrence, candidate_words,weights)
                counter=counter+1
                count=count+score
                #print("score= ",score)
                #f = open('bleu.txt', 'w')
                #f.write(score)
                #f.write('\n')
                if counter%1000==0:
                    print("bleu at "+str(counter)+" images= "+str(count/counter))
            BLEU=count/len(dictionary)
            print("BLEU= ",BLEU)





FilePath1 = "annotations/captions_train2014.json"
FilePath2 = "annotations/captions_val2014.json"
calculate_BLEU(FilePath2)



