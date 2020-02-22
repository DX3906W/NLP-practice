
import numpy as np
import jieba
import jieba.posseg as pseg
from tkinter import _flatten

class TextRank(object):

    def __init__(self, dataSet, window, alpha, epoch):

        self.dataSet = dataSet
        self.window = window
        self.alpha = alpha
        self.edge_dict = {} #记录节点的边连接字典
        self.word_data = []
        self.word_list = []
        self.word_set = set()
        self.epoch = epoch
        self.word_index = {}
        self.index_word = {}

    #split sentence into words
    def splitSentence(self):

        jieba.load_userdict('user_dict.txt')
        tag_filter = ['a', 'd', 'n', 'v']

        for sentence in self.dataSet:
            seg_result = pseg.cut(self.sentence)
            self.word_data.append([s.word for s in seg_result if s.flag in tag_filter])

        self.word_list = list(set(self.word_data))

    #build edge between words by giving window
    def createNodes(self):

        for words in self.word_data:
            for index, word in enumerate(words):
                left = index - self.window + 1 if index >= self.window - 1 else 0
                right = index + self.window if index + self.window <= len(words) else len(words)

                self.edge_dict[word] = list(set(self.edge_dict[word].append(words[left, right])))


    #create a matrix to represent the relationship between each two words
    def createMatrix(self):

        self.matrix = np.zero([len(self.word_list), len(self.word_list)])

        for index, word in self.word_list:
            self.word_index[word]  = index
            self.index_word[index] = word

        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1

        #out
        self.matrix /= np.sum(self.matrix)

    def calculatePR(self):

        self.PR = np.ones([len(self.word_list), 1])

        for i in range(self.epoch):
            self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)

    def getResult(self):

        self.word_PR = {}

        for index, word in self.word_list:
            self.word_PR[word] = self.word_PR[index][0]

        res = sorted(self.word_PR.items(), key=lambda x:  x[1], reverse=True)

        print(res)


if __name__=="__main__":
    s = '程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
    tr = TextRank(s, 3, 0.85, 700)
    tr.cutSentence()
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    tr.printResult()

