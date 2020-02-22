
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class tokenizer(object):

    def __init__(self, dataSet, do_lower_case=True):

        self.dataSet = dataSet


    def load_vocab(self, text):
        self.cut(text)

    def cut(self, text):
        vocab_list = []
        for sentence in text:
            vocab_list.append(nltk.word_tokenize())

        vocab_list = self.stopWord(vocab_list)

        return vocab_list

    def stopWord(self, vocab_list):

        vocab_list_ = []

        stop_words = set(stopwords.words('english'))
        for words in vocab_list:
            words_ = [w for w in words if vocab_list not in stopwords]
            self.stemming(words)
            vocab_list_.append(words_)

        return vocab_list_

    def stemming(self, vocab_list):

        vocab_list_ = []

        porter_stemmer = PorterStemmer()
        for words in vocab_list:
            vocab_list_.append(porter_stemmer.stem(words))

        return vocab_list_


