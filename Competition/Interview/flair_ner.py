from flair.data import Sentence
from flair.models import SequenceTagger
from extract_pdf import get_data


data = get_data()

for index, file in enumerate(data):
  doc = ' '.join(txt_data)
  # make a sentence
  sentence = Sentence(doc)
  # load the NER tagger
  tagger = SequenceTagger.load('ner')
  tagger.predict(sentence)
  # iterate over entities and print
  result = sentence.get_spans('ner')
  with open('F:/mygit/NLP-practice/Competition/Interview/output/flair_ner_'+str(index)+'.txt', 'w', encoding='utf-8') as f:
    for entity in result:
      f.write(entity.text + ', ' + entity.tag + '\n')
