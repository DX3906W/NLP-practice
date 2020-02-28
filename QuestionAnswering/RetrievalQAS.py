
import json
import jieba
from nltk.corpus import stopwords
import gensim
import numpy as np

def load_data():
    path = '../dataSet/QAS/train-v2.0.json'
    file = open(path)
    json_data = json.load(file)

    qlist = []
    alist = []

    for item in json_data['data']:
        for i in item['paragraphs']:
            for j in i['qas']:
                if(j['question']):
                    qlist.append(j["question"])
                else:
                    qlist.append(-1)
                if j['answers']:
                    alist.append(j["answers"][0]['text'])
                else:
                    alist.append(-1)

    return qlist, alist

def segmentation(qlist):

    processedQlist = []

    if is instance(qlist):
        for q in qlist:
            processedQlist.append(jieba.cut(q))

    else:
        processedQlist.append(jieba.cut(q))

    processedAlist = remove_stop(processedQlist)

    return processedAlist


def remove_stop(qlist):

    stop_word = set(stopwords.words('chinese'))
    processedQlist = []

    if isinstance(qlist[0], list):
        for question in qlist:
            processedQlist.append([x for x in question if x not in stop_word])
    else:
        processedQlist = [x for x in qlist if x not in stop_word]

    return processedQlist


def preprocess(mode='train', input_question):

    if mode=='train':
        qlist, alist = load_data()

        qlist = segmentation(qlist)

        qlist = remove_stop(qlist)

        return qlist, alist

    elif mode=='test':
        words = segmentation(input_question)
        words = remove_stop(words)

        return qlist

    else:
        raise Exception("Invilid mode: ", mode)


def builg_dict(qlist):

    word_list = list(set(np.array(qlist).flatten().tolist()))

    word_embeddings = wprd_embedding('word2vec', word_list)

    word_dict = dict(word_list, wor_embedding)

    return word_embeddings

def word_embedding(mode='glove', word_list):

    if mode=='glove':
        print()
    elif mode=='word2vec':
        word_embeddings = []
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format('../datadSet/zhwiki_2017_03.sg_50d.word2vec',binary=Ture)
        for word in word_list:
            word_embeddings.append(word_vectors.most_similar(word))

            return word_embeddings
    else:
        raise Exception('Invilid mode: ', mode)

# word_vectors=gensim.models.KeyedVectors.load_word2vec_format('../dataSet/zhwiki_2017_03.sg_50d.word2vec',binary=False)
# sim = word_vectors.most_similar(u'蛋白质',topn=10)
# for i in sim:
#     print(i)
def sen2vec():
    qlist, alist = preprocess()
    word_dict = bulid_dict(qlist)
    questionVectors = {}

    for q, a in qlist, alist:
        wordVectors = []
        for w in q:
            wordVectors.append(word_dict[w])

        questionVector = np.array(wordVectors).mean(axis=0)
        questionVectros[questionVector] = a

    return questionVectors


def similarity(input_question, c_question):

    input_question = np.array(input_question)
    c_question = np.array(c_question)

    num = input_question * c_question
    denom = np.linalg.norm(input_question) * np.linalg.norm(input_question)

    sim = num / denom

    return sim


def retrival(input_question):

    answer = ''
    max = 0

    QApair = sen2vec()

    input_words = preprocess(mode='test', input_question)
    word_dict = build_dict()

    input_vector = []
    for word in input_words:
        input_vector.append(word_dict[word])

    input_vertor = np.array(input_vector).mean(axis=1)

    for question in questionVectors.keys():
        if similarity(input_vertor, question)>max:
            answer = QApair[question]

    return answer
