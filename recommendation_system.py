import json
from difflib import SequenceMatcher

#take caption from json file
def getAdds(caption):
        FilePath = "videos_infos.json"
        nratio=0
        oratio=0
        title=" "
        link=" "
        with open(FilePath) as json_data: #by default close file (with)
             Properties=json.load(json_data)
             for record in Properties["Info"]:
                 tag=record['tags']
                 nratio = SequenceMatcher(None, caption, tag).ratio()
                 if nratio>oratio:
                     oratio=nratio
                     title=record['title']
                     link=record['url']
             if nratio==0:
                 title="default advertisement"
                 link="https://www.youtube.com/watch?v=PZguUhAB_hI"



        return title,link

title,link=getAdds("woman bab hous sk")
print(title,link)
