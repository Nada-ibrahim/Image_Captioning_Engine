import json
import pickle
import re
from pathlib import Path

FilePath1="annotations/captions_train2014.json"
FilePath2="annotations/captions_val2014.json"

Properties={}
def additional_words(dictionaryWords):
     add_words={'red','green','orange','yellow','golden','silver','brown','black','white','blue','purple','pink','color','sports','skirt','jacket','coat','trouser','short','suit','dress','shoes','sweaters','t-shirt','accessories','fashion'}
     for w in add_words:
         if w not in dictionaryWords:
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
    dict_path = Path("obj/dictionary")
    dictionary = {}
    reversedDictionary = {}
    if not dict_path.exists():
        txt1 = extractData(FilePath1)
        txt2 = extractData(FilePath2)
        fullText = txt1.lower() + txt2.lower()  # convert annotations files json to text file
        del txt1
        del txt2
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
        dictionaryWords=additional_words(dictionaryWords)
        dictionaryWords = standardWords + list(set(dictionaryWords))

        listOfInt = [i for i in range(0, len(dictionaryWords))]
        zipbObj = zip(dictionaryWords, listOfInt)
        dictionary = dict(zipbObj)
        reversedDictionary = {i: dictionaryWords[i] for i in range(0, len(dictionaryWords))}

        with open("obj/dictionary", "wb") as f:
            pickle.dump(dictionary, f)
            pickle.dump(reversedDictionary, f)
    else:
        with open("obj/dictionary", "rb") as f:
            dictionary = pickle.load(f)
            reversedDictionary = pickle.load(f)
    return dictionary, reversedDictionary


def encode_annotations(annotations_path,saved_path, dictionary):
    # print(dictionary)
    path = Path(saved_path)
    if not path.exists():
        oneCaption = " "
        maxlength=0
        startTag=dictionary["<start>"]
        endTag=dictionary["<end>"]

        with open(annotations_path) as json_data:  # by default close file (with)
            Properties = json.load(json_data)

            for record in Properties["annotations"]:
                oneCaption=record['caption'].lower()
                word = re.findall(r"[\w']+", oneCaption)
                length=word.__len__()
                if maxlength <length:
                    maxlength=length
                    print(maxlength)

                unkonwnValues = [i for i in word if dictionary.get(i) == None]
                counter = -1
                for i in word:
                    counter += 1
                    if unkonwnValues.count(i) > 0:
                        word[counter] = '<unknown>'

                newText = []
                for i in word:
                    newText.append(dictionary.get(i))
                newText.insert(0,startTag)
                newText.append(endTag)
                file = open(saved_path, "a")
                newText = ' '.join(str(e) for e in newText)
                file.write(newText)
                file.write("\n")
                file.close()



