'''
论文原文《Applying Deep Learning To Answer Selection- A Study And An Open Task》
'''


import json
import jieba
from nltk.corpus import stopwords
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

    processedList = []

    for q in qlist:
        processedList.append(jieba.cut(q))

    processedList = remove_stop(processedList)

    return processedList

def remove_stop(qlist):

    stop_word = set(stopwords.words('chinese'))
    processedList = []

    for q in qlist:
        processedList.append([x for x in q if x not in stop_word])

    return processedList

def build_dict(qlist):

    word_list = list(set(np.array(qlist).flatten().tolist()))

    word_embeddings = word_embedding(word_list)

    word_dict = dict(word_list, word_embeddings)

    return word_dict


def word_embedding(word_list):

    word_embeddings = []

    # word_vectors = gensim.models.KeyedVectors.load_word2vec_format('../datadSet/zhwiki_2017_03.sg_50d.word2vec', binary=True)
    # for word in word_list:
    #     print(word_vectors.most_similar(word).shape)
    #     word_embeddings.append(word_vectors.most_similar(word))

    return word_embeddings


def sen2vec(qlist, word_dict):

    QVectors = []

    for q in qlist:
        temp = []
        for word in q:
            temp.append(word_dict[word])
        QVectors.append(temp)

    return QVectors


def preprocess():
    qlist, alist = load_data()

    processedQlist = segmentation(qlist)
    processedAlist = segmentation(alist)

    corpus = [].append(processedAlist).append(processedQlist)

    word_dict = build_dict(corpus)

    shuf = np.arange(len(qlist))

    Qpvectors = sen2vec(processedQlist, word_dict)
    Qnvectors = Qpvectors[shuf, :, :]
    Avectors = sen2vec(processedAlist, word_dict)

    return Qpvectors, Qnvectors, Avectors

