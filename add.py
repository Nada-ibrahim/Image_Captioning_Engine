
dictionaryWords={'red', 'green', 'orange','rawan','habl','try','yarab','clothess','quran','fatakat'}

def additional_words(dictionaryWords):
    add_words = {'red', 'green', 'orange', 'yellow', 'golden', 'silver', 'brown', 'black', 'white', 'blue', 'purple',
                 'pink', 'color', 'sports', 'skirt', 'jacket', 'coat', 'trouser', 'short', 'suit', 'dress', 'shoes',
                 'sweaters', 't-shirt', 'accessories', 'fashion'}
    print(len(add_words))
    for w in add_words:
        if w not in dictionaryWords:
            newItem = {w}
            dictionaryWords.update(newItem)

    return dictionaryWords
print("before")
print(len(dictionaryWords))
additional_words(dictionaryWords)
print("after")
print(len(dictionaryWords))
