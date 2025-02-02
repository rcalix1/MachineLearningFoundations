#!/usr/bin/env python
# coding: utf-8

# In[1]:


## simple binary log reg 

import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn import datasets
import matplotlib.pyplot as plt


#######################################

def accuracy(y_true, y_pred):
    accuracy = np.sum( y_true == y_pred ) / len(y_true)
    return accuracy


#######################################

def mySigmoid(z):
    return 1 / (1 + np.exp(-z))

#######################################

def predict(X, weights, bias):
    z = np.dot( X, weights ) + bias
    y_predicted = mySigmoid(z)
    y_pred = [  1 if i > 0.5 else 0 for i in y_predicted  ]
    return np.array(   y_pred   )
 
#######################################

def fit(X, y):
    lr = 0.0001
    n_iters = 1000
    n_samples, n_features = X.shape  
    
    weights = np.zeros(n_features)
    bias = 0
    
    # gradient descent
    for _ in range(n_iters):
        z = np.dot(   X, weights   ) + bias
        y_pred = mySigmoid(z)
        
        ## compute gradients
        ## derivatives of cross entropy
        dw = (1 / n_samples) * np.dot(  X.T, (y_pred - y)  )
        db = (1 / n_samples) * np.sum(  y_pred - y  )
        
        ## update parameters using the gradients
        weights = weights - lr * dw
        bias    = bias - lr * db
        
    return weights, bias


#######################################

bc = datasets.load_breast_cancer() 
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#######################################

print(X.shape) 
print(y.shape)

#######################################

#fig = plt.figure(figsize=(8, 6))
#plt.scatter(X[:, 0], y, color = 'b', marker = 'o', s = 30)
#plt.show()

######################################


weights, bias  = fit(X_train, y_train)
y_pred = predict(X_test, weights, bias)

accu_r = accuracy(y_pred, y_test)
print("accuracy ", accu_r)

print(y_pred[:40])
print(y_test[:40])

#####################################

print("<<<<<<<<<<<<DONE>>>>>>>>>>>>>")



# In[ ]:




