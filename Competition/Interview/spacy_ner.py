import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import en_core_web_md
import en_core_web_lg

import appConfig
from extract_pdf import get_data

data = get_data()
for  i in range(3):
    if i==0:
        nlp = en_core_web_sm.load()
        file_name = 'spacy_small_ner'
    elif i==1:
        nlp = en_core_web_md.load()
        file_name = 'spacy_mid_ner'
    else:
        nlp = en_core_web_lg.load()
        file_name = 'spacy_large_ner'
    for index, file in enumerate(data):
        doc = nlp(file)
        with open(appConfig.out_file_path+file_name+str(index+1)+'.txt', 'w') as f:
            for X in doc.ents:
                f.write(X.text+','+X.label_+'\n')

        labels= [x.label_ for x in doc.ents]
        print(Counter(labels))
