## use conda prompt
## conda create -n tf_v2 tensorflow
## conda env list
## conda activate tf_v2
## pip install keras, sklearn

#################################################################

import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import tensorflow_datasets as tfds

##################################################################
'''
iris = tfds.load('iris')
X_train_iris = iris['train']
print(X_train_iris.shape)
'''
##################################################################

tf.random.set_seed(1)
np.random.seed(1)

##################################################################

X = np.random.uniform(  low=-1, high=1, size=(200, 2)   )
y = np.ones(   len(X)    ) 

## XOR data

y[  X[:, 0] * X[:, 1] < 0   ] = 0


print(y.shape)
print(X.shape)

##################################################################

X_train = X[:100, :]
y_train = y[:100]

X_test  = X[100:, :]
y_test  = y[100:]

##################################################################
## logistic regression
## no hidden layers

model = tf.keras.Sequential()
model.add(
             tf.keras.layers.Dense(
                                      units=1,
                                      input_shape=(2, ),
                                      activation='sigmoid'
                                      
             
                                  )

          )
model.summary()

model.compile(
                 optimizer=tf.keras.optimizers.SGD(),
                 loss = tf.keras.losses.BinaryCrossentropy(),
                 metrics=[tf.keras.metrics.BinaryAccuracy()]

             )
             
'''
history = model.fit(
                       X_train,
                       y_train,
                       validation_data=(X_test, y_test),
                       epochs=200,
                       batch_size=2,
                       verbose=1

                   )
'''

##################################################################
## multi-layer perceptron


##################################################################
## DNN
## deep neural net with 3 hidden layers

model = tf.keras.Sequential()
model.add(  tf.keras.layers.Dense(units=4, input_shape=(2,), activation='relu'))
model.add(  tf.keras.layers.Dense(units=4, activation='relu'))
model.add(  tf.keras.layers.Dense(units=4, activation='relu'))
model.add(  tf.keras.layers.Dense(units=1, activation='sigmoid'))


model.summary()

model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[   tf.keras.metrics.BinaryAccuracy()   ]

             )



hist = model.fit(
                    X_train,
                    y_train,
                    validation_data = (X_test, y_test),
                    epochs=200,
                    batch_size=2,
                    verbose=1

                )


##################################################################

print("<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>")

