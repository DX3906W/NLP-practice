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

# nltk.download('stopwords')
# nltk.download('punkt')

class TF_IDF():

    def __init__(self, dataSet):

        self.dataSet = dataSet
        self.count = []
        self.tf = []
        self.idf = []
        self.tf_idf = []

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

    def TF(self):
        for c in self.count:
            record = {}
            for w in c.keys():
                record[w] = c[w] / sum(c.values())

            self.tf.append(record)

    def IDF(self):
        for c in self.count:
            record = {}
            for w in c.keys():
                record[w] = sum(1 for c_ in self.count if w in c_)

            self.idf.append(record)


    def getResult(self):

        self.dataPreprocess()

        self.TF()

        self.IDF()

        for index, c in enumerate(self.count):
            record = {}
            for k in c.keys():
                record[k] = self.tf[index][k] / self.idf[index][k]

            record = dict(sorted(record.items(), key=lambda x: x[1], reverse=True))
            self.tf_idf.append(record)

        return self.tf_idf

if __name__ == "__main__":

    text_1 = "In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. Tf–idf is one of the most popular term-weighting schemes today; 83% of text-based recommender systems in digital libraries use tf–idf."
    text_2 = "Variations of the tf–idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. tf–idf can be successfully used for stop-words filtering in various subject fields, including text summarization and classification."
    text_3 = "One of the simplest ranking functions is computed by summing the tf–idf for each query term; many more sophisticated ranking functions are variants of this simple model."

    tf = TF_IDF([text_1, text_2, text_3])

    result = tf.getResult()

    print(result)
