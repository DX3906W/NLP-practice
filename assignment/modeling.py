
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class KNN_Model():

    def __init__(self):
        self.train = []
        self.test = []
        self.LABEL_COLUMN = 'Label'
        self.LABELS = [0, 1, 2]
        self.FILEPATH = r"F:\assignment\data\dataSet.csv"

    def get_dataset(self):

        df = pd.read_csv(self.FILEPATH)
        data = np.array(df)

        dataSize = len([k[0] for k in data])

        random.shuffle(data)

        train = data[ : int(0.7 * dataSize)]
        test = data[int(0.7 * dataSize) : ]

        train = pd.DataFrame(columns=["Mora", "Karlstad", "Gavle", "Vasteras", "Label"], data=[row[1: ] for row in train])
        test = pd.DataFrame(columns=["Mora", "Karlstad", "Gavle", "Vasteras", "Label"], data=[row[1: ] for row in test])

        return  train[["Mora", "Karlstad", "Gavle", "Vasteras"]], np.array(train[["Label"]]), np.array(test[["Mora", "Karlstad", "Gavle", "Vasteras"]]), np.array(test[["Label"]])

    def overview(self):
        train_x, train_y, test_x, test_y = self.get_dataset()

        accuracy = []

        for k in range(5, 21):
            knn_classifier = KNeighborsClassifier(k, weights='distance')
            knn_classifier.fit(train_x, train_y.ravel())
            y_predict = knn_classifier.predict(test_x)
            scores = knn_classifier.score(test_x, test_y.ravel())

            accuracy.append(scores)

            print('k: {}, acc: {}'.format(k, np.sum(y_predict == test_y.ravel()) / len(test_x)))

        plt.plot(range(5, 21), accuracy, linewidth=3)

        plt.title("The accuracy in terms of different k", fontsize=24)

        plt.xlabel("K", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)

        plt.show()
        plt.show()

    def predict(self, test_x):

        train_x, train_y, _, _ = self.get_dataset()

        prediction = []

        for k in range(5, 21):
            knn_classifier = KNeighborsClassifier(k, weights='distance')
            knn_classifier.fit(train_x, train_y.ravel())
            y_predict = knn_classifier.predict(test_x)

            prediction.append(y_predict)

            print('k: {}, prediction: {}'.format(k, y_predict))

        return prediction
