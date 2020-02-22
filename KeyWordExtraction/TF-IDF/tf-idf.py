import nltk
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from nltk.stem.porter import *
import numpy as np
import re
import pandas as pd


class TF_IDF(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet
        self.count = []
        self.tf = []
        self.idf = []
        self.tf_idf = []

        self.dataPreprocess()

        self.tf()

        self.idf()


    def dataPreprocess(self):
        for article in self.dataSet:

            tokens = self.cutSentence(article)
            tokens = self.removeStopWords(tokens)

            self.count.append(Counter(tokens))

    def cutSentence(self, sentence):
        lower_text = sentence.lower()
        s = re.sub(r'[^\w\s]', '', lower_text)
        tokens = nltk.word_tokenize(s)

        return tokens

    def removeStopWords(self, tokens):
        stop_words = set(stopwords.words('english'))
        tokens = [i for i in tokens if i not in stop_words]

        return tokens

    def tf(self):
        for c in self.count:
            record = {}
            for w in c.keys:
                record[w] = c[w] / sum(c.values)

            self.tf.append(record)

    def idf(self):
        for c in self.count:
            record = {}
            for w in c.keys:
                record[w] = sum(1 for c_ in self.count if w in c_)

            self.idf.append(record)


    def getResult(self):
        for index, c in enumerate(self.count):
            record = {}
            for k in c.keys:
                record[k] = self.tf[index][k] / self.idf[index][k]

            record = dict(sorted(record.items(), key=lambda x: x[1], reverse=True))
            self.tf_idf.append(record)

if __name__ == "__main__":

    tf = TF_IDF(["", ""])

    tf.getResult()



