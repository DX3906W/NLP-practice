from stanfordcorenlp import StanfordCoreNLP
from extract_pdf import get_data

nlp = StanfordCoreNLP(r'E:\stanfordnlp',lang='en')

data = get_data()

for index, file in enumerate(data):
    ner_tag = nlp.ner(file)
    with open('F:/mygit/NLP-practice/Competition/Interview/output/stanfordnlp_ner'+str(index+1)+'.txt', 'w') as f:
        for word in ner_tag:
            f.write(word[0]+', '+word[1]+'\n')
