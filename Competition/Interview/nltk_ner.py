import sys
import nltk
from extract_pdf import get_data
from nltk.corpus import treebank

data = get_data()
for index, text in enumerate(data):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged, binary=True)

    entities_str = str(entities)

    with open('output/nltk_ner_' + str(index+1) + '.txt', 'w') as f:
        f.write(entities_str)
