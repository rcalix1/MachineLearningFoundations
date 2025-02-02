#!/usr/bin/env python
# coding: utf-8

# In[1]:



## knn
import numpy as np
from collections import Counter

k = 5

######################################################

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum(     (v1 - v2) ** 2    ))

######################################################

X_train = ??    ##   [ 30, [f1, f2, f3, ..., fn]]
y_train = ??

## X_test  [50, 4]
test_x = ??     ##  [f1, f2, f3, ..., fn]

######################################################

def predict(test_x):
    ## calculate distances between test_x and all data samples in X
    distances = [ euclidean_distance(test_x, x )  for x in X_train   ]
    ## distances is a vector of 30 distances 
    
    ## distances [23, 2,  145, 23  , 5,   17 , 890, ....]  =>>  []
               
    
    ## sort by distance and return the k closest neighbors
    ## argsort returns the indices of the k nearest neighbors
    k_neighbor_indeces =   np.argsort(   distances   )[:k]
    
    ## extract labels from y_train
    labels = [    y_train[i]  for i in k_neighbor_indeces   ]
    ## imagine labels = [1, 1, 1, 0, 1]
    

    ##select the most common label in labels
    most_common_label = Counter(labels).most_common(1)

    return most_common_label

####################################################################


for test_x in X_test:
    print(    predict(test_x)   )


# In[ ]:




