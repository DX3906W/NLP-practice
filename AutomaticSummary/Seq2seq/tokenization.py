import nltk
import re
from nltk.tokenize import word_tokenize
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class tokenizator:
    def __init__(self， lowCase=True, removeStopWord=True, stem=True):
        self.lowCase = lowCase
        self.removeStopWord = removeStopWord
        self.stem = stem

        self.train_article_path = "sumdata/train/train.article.txt"
        self.train_title_path = "sumdata/train/train.title.txt"
        self.valid_article_path = "sumdata/train/valid.article.filter.txt"
        self.valid_title_path = "sumdata/train/valid.title.filter.txt"

        train_article_list, train_title_list, valid_article_list, valid_title_list = loadData()




    def loadData(self, path):
        train_article = open(self.train_article_path, 'r')
        train_title = open(self.train_title_path, 'r')
        valid_article = open(self.valid_article_path, 'r')
        valid_title = open(self.valid_title_path, 'r')

        train_article_list = [normalization(sentence) for sentence in train_article.readlines()]
        train_title_list = [normalization(sentence) for sentence in train_title.readlines()]
        valid_article_list = [normalization(sentence) for sentence in valid_article.readlines()]
        valid_title_list = [normalization(sentence) for sentence in valid_title.readlines()]

        return train_article_list, train_title_list, valid_article_list, valid_title_list

    def normalization(self, sentence):

        sentence = clearStr(sentence)

        word_list = removeStopWord(sentence)

        word_list = stemming(word_list)

        return word_list

    def clearStr(self, sentence):

        sentence = re.sub("[#.]+", "#", sentence)
        return sentence

    def removeStopWord(self, sentence):
        stop_words = set(stopwords.words(‘english’))
        word_list = word_tokenize(sentence)

        word_list = [x for x in word_list: if x not in stop_words]

        return word_list

    def stemming(self, word_list):

        words = []
        for word in word_list:
            words.append(nltk.PorterStemmer().stem_word(word))
        return words

    def buildDataSet(self, action, word_list):
        word_dict = {}
        if action == 'train':
            word_dict["<s>"] = 0
            word_dict["<pad>"] = 1
            word_dict["<unk>"] = 2
            word_dict["</s>"] = 3
            word_set = build_dict(word_list).toList()
            for word, index n enumerate(word_set):
                word_dict(word) = index+4
            with open["word_dict.pickle", "wb"] as f:
                pickle.dump(word_dict, f)

        elif step == "valid":
            with open("word_dict.pickle", "rb") as f:
                word_dict = pickle.load(f)


    def build_dict(self, word_list):
        word_dict = set()
        for words in word_list:
            word_dict.add(words)

        return word_set.toList()

    def getData():

        return self.train_data, self.valid_data
