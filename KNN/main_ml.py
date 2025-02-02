#!/usr/bin/env python
# coding: utf-8

# In[2]:



## my_main ML function

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

## from myKNN import myKNN    ## This is your own file 

###################################################################

def accuracy(y_test, y_pred):
    accuracy_value = np.sum(y_true == y_pred) / len(y_test)
    return accuracy_value


###################################################################

## 150 samples, 4 features, 3 classes 
iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## scale the data 

####################################################################

k = 5
myMl_knn = myKNN(k=k)

myML_knn.train(X_train, y_train)
y_pred = myML_knn.predict(X_test)

####################################################################


print("knn accuracy results: ")

accuracy_result = accuracy(y_test, y_pred)

print(accuracy_result)


# In[ ]:




