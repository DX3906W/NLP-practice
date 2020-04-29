from extract_pdf import get_data
from nltk.tag.stanford import StanfordNERTagger
import nltk

# nlp = StanfordCoreNLP(r'E:\stanfordnlp',lang='en')
jar = 'E:/stanfordnlp_ner/stanford-ner.jar'
model = 'E:/stanfordnlp_ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
data = get_data()

for index, file in enumerate(data):
    ner_tagger = StanfordNERTagger(model, jar)
    words = nltk.word_tokenize(file)
    ner_tags = ner_tagger.tag(words)
    with open('F:/mygit/NLP-practice/Competition/Interview/output/stanfordnlp_ner'+str(index+1)+'.txt', 'w', encoding='utf-8') as f:
        for word in ner_tags:
            if word[1] != 'O':
                f.write(word[0]+', '+word[1]+'\n')
