import numpy as np
import pandas as pd
import seaborn as sns
import config

def read_data():
    train_data = pd.read_csv(config.train_dir)
    test_data = pd.read_csv(config.test_dir)

    train_len = [len(i) for i in train_data['text'].values]
    test_len = [len(i) for i in test_data['text'].values]

    sns.kdeplot(train_len)
    sns.kdeplot(train_len)

    labels = train_data['label'].values
    sns.kdeplot(labels)

    print("djfbm")

read_data()
