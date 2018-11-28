import json
import re
FilePath1="captions_train2014.json"
FilePath2="captions_val2014.json"

Properties={}

#take caption from json file
def extractData(FilePath):
        txt = " "
        with open(FilePath) as json_data: #by default close file (with)
             Properties=json.load(json_data)
             for record in Properties["annotations"]:
                 txt+= record['caption']
        return txt

def get_dictionaries(annotations_path):
    dictionary = {}
    reversedDictionary = {}
    txt1 = extractData(FilePath1)
    txt2 = extractData(FilePath2)
    fullText = txt1 + txt2 #convert annotations files json to text file

    print("extract& save")

    word=re.findall(r"[\w']+", fullText) #splitting
    allWords = {}
    for w in word:
        if w in allWords.keys():
            value = allWords[w]
            allWords[w] = value + 1
        else:
            newItem = {w: 1}
            allWords.update(newItem)
    standardWords=[ "<start>","<end>","unKnown"]
    dictionaryWords = [i for i in allWords if allWords[i] > 5]
    dictionaryWords =standardWords+ list(set(dictionaryWords))


    listOfInt=[ i for i in range(0, len(dictionaryWords)) ]
    zipbObj = zip(dictionaryWords, listOfInt)
    dictionary =dict(zipbObj)
    reversedDictionary = { i : dictionaryWords[i] for i in range(0, len(dictionaryWords) ) }

    return dictionary,reversedDictionary


def encode_annotations(annotations_path,saved_path):
    print(dictionary)
    oneCaption = " "
    maxlength=0
    startTag=dictionary["<start>"]
    endTag=dictionary["<end>"]

    with open(annotations_path) as json_data:  # by default close file (with)
        Properties = json.load(json_data)

        for record in Properties["annotations"]:
            oneCaption=record['caption']
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
                    word[counter] = 'unKnown'

            newText = []
            for i in word:
                newText.append(dictionary.get(i))
            newText.insert(0,startTag)
            newText.append(endTag)
            file = open(saved_path, "a")
            newText = ' '.join(str(e) for e in newText)
            file.write(newText)
            file.write(" /n ")
            file.close()



    print(maxlength)

    return maxlength;

annotations_path=" "

dictionary, reversedDictionary = get_dictionaries(annotations_path)
maxlength_train=encode_annotations(FilePath1,'encoded_train_annotations.txt')
maxlength_val=encode_annotations(FilePath2,'encoded_val_annotations.txt')
