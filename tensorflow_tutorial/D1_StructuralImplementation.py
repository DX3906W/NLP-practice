import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import models, layers


train_df = pd.read_csv('data/titanic/train.csv')
test_df = pd.read_csv('data/titanic/test.csv')

# print(train_df.columns.to_list())
# print(train_df.head(5))

#the distribution of label
# ax = train_df['Survived'].value_counts().plot(kind='bar',
#                                               figsize=(12, 8),
#                                               fontsize=12)
# ax.set_xlabel('labels', fontsize=15)
# ax.set_ylabel('count', fontsize=15)
#
# plt.show()

# the distribution of age
# ax = train_df['Age'].plot(kind='hist',
#                           bins=20,
#                           figsize=(12, 8),
#                           fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# ax.set_ylabel('Frequency', fontsize=15)
# plt.show()

#The relevance of label and age
# ax = train_df.query('Survived == 0')['Age'].plot(kind = 'density',
#                       figsize = (12,8),fontsize=15)
# train_df.query('Survived == 1')['Age'].plot(kind = 'density',
#                       figsize = (12,8),fontsize=15)
# ax.legend(['Survived==0','Survived==1'],fontsize = 12)
# ax.set_ylabel('Density',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()

def data_preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)

x_train = data_preprocessing(train_df)
y_train = train_df['Survived'].values

x_test = data_preprocessing(test_df)
y_test = test_df['Survived'].values

print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', x_test.shape)

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_split=0.2)

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history, "acc")

model.evaluate(x_test, y_test)

model.predict_classes(x_test, batch_size=64)

model.save('model/keras_model.h5')

del model

# save model and weight
model = models.load_model('model/keras_model.h5')
model.evaluate(x_test, y_test)

# save model architecture in json format
json_str = model.to_json()
model_json = models.model_from_json(json_str)

# save model weight
model.save_weights('model/kerass_weight.h5')

# use saved model
model_json = models.model_from_json(json_str)
model_json.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc']
)

model_json.load_weights('model/kerass_weight.h5')
model_json.evaluate(x_test, y_test)


