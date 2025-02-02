#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

########################################
## for 2 class only [0, 1]


def fit( X_train, y_train ):
    learning_rate = 0.01
    n_samples, n_features = X.shape
    ## y = w1*x1 + w2*x2 + ... + b
    
    weights =   np.zeros(  n_features  )    ## [   0   0    0   0     0    0  ...   0]
    bias    =   0
    
    y_ = np.array(       [ 1 if i > 0 else 0 for i in y_train  ])
    #y_ = y_train
    #y_ = np.array(       [ 1 if i > 0 else -1 for i in y_train  ])
    
    
    print(y_)
    
    for _ in range(1000):
        
        for idx, x_i in enumerate(X_train):
            
            linear_output  = np.dot(  x_i,  weights   ) + bias       ## y = w*x + b
            y_pred         = activation_function(linear_output)
            
            ## compare predicted with real value
            ## Perceptron update rule
            
            update = learning_rate * (   y_[idx]  -   y_pred  )
            
            weights = weights + update * x_i      ##    [0  0 0 0 0 0 0 ]  +  [x1  x2  x3   x4  ...]*update
            bias = bias + update
    
    return weights, bias
                 
            
            
            
########################################

def activation_function(x):
    return np.where(    x>=0, 1, 0    )
    
    
########################################
    
def predict(X_test, weights, bias):
    linear_output = np.dot(X_test, weights) + bias          ##  y = w*x + b 
    y_pred = activation_function(    linear_output    )
    return y_pred


#######################################

def accuracy(y_true, y_pred):
    accuracy = np.sum(   y_true == y_pred    ) / len(   y_true   )
    return accuracy
    
#######################################

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#######################################

print(X[:10, :])
print(y[:10])

#######################################

weights, bias = fit(X_train, y_train)
y_pred = predict(X_test, weights, bias)


#######################################

## compare y_pred to y_test

print(      accuracy(y_test, y_pred)    )

print("*********************************")
print(y_test[:20])
print(y_pred[:20])
    


# In[ ]:




