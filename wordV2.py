import json
import pickle
import re
from pathlib import Path

FilePath1="E:\FCI\GP\Code\captions_train2014.json"
FilePath2="E:\FCI\GP\Code\captions_val2014.json"

Properties={}
def additional_words(dictionaryWords):
     add_words={'red','green','orange','yellow','golden','silver','brown','black','white','blue','purple','pink','color','sports','skirt','jacket','coat','trouser','short','suit','dress','shoes','sweaters','t-shirt','accessories','fashion'}
     for w in add_words:
         if w not in dictionaryWords:
             print(w)
             dictionaryWords.append(w)

     return dictionaryWords


#take caption from json file
def extractData(FilePath):
        txt = " "
        with open(FilePath) as json_data: #by default close file (with)
             Properties=json.load(json_data)
             for record in Properties["annotations"]:
                 txt+= record['caption'] + " "
        return txt


def create_dictionaries():
    dictionary = {}
    reversedDictionary = {}
    print("here")
    txt1 = extractData(FilePath1)
    txt2 = extractData(FilePath2)
    fullText = txt1.lower() + txt2.lower()  # convert annotations files json to text file
    del txt1
    del txt2
    print("hereee2")
    print("extract& save")

    word = re.findall(r"[\w']+", fullText)  # splitting
    allWords = {}
    for w in word:
        if w in allWords.keys():
            value = allWords[w]
            allWords[w] = value + 1
        else:
            newItem = {w: 1}
            allWords.update(newItem)
    standardWords = ["<start>", "<end>", "<unknown>"]
    dictionaryWords = [i for i in allWords if allWords[i] > 5]
    print(len(dictionaryWords))
    dictionaryWords=additional_words(dictionaryWords)
    print("after")
    print(len(dictionaryWords))
    dictionaryWords = standardWords + list(set(dictionaryWords))
    print(dictionaryWords)

    listOfInt = [i for i in range(0, len(dictionaryWords))]
    zipbObj = zip(dictionaryWords, listOfInt)
    dictionary = dict(zipbObj)
    reversedDictionary = {i: dictionaryWords[i] for i in range(0, len(dictionaryWords))}

    return dictionary, reversedDictionary



create_dictionaries()


