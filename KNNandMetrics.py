#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score

import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder

#############################################################

k = 5

#############################################################
## from myKNN import myKNK


def euclidean_distance(v1, v2):
    return np.sqrt(np.sum(     (v1 - v2) ** 2    ))


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



# In[2]:


## dataset: Iris 150 samples, 4 features, 3 classes


# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# print(X)

#######################################################
'''

f_numpy = open('iris.csv', "r")
Matrix_data = np.loadtxt(f_numpy, delimiter=',', skiprows=1)
print(   Matrix_data  )

'''

#####################################################

df = pd.read_csv('iris.csv', header=None)

#print(   df    )

X = df.loc[1:, :3].values
X = X.astype(float)


y = df.loc[1:,  4 ].values

#print(   y   )

le = LabelEncoder()
y = le.fit_transform(   y   )      ## changes a label string into integers

#print(  X  )
#print(  y  )

 


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42 )


# In[4]:


print(y_test)
print(y_train)
print(  X_train   )
print(  X_test    )


# In[5]:



def accuracy(y_pred, y_test):
    accuracy_value = np.sum(y_pred == y_test) / len(y_test)
    return accuracy_value



# In[6]:




def print_stats_percentage_train_test(algorithm_name, y_test, y_pred):    
     print("------------------------------------------------------")
     print("------------------------------------------------------")
    
     print("algorithm is: ", algorithm_name)
        
     print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
     
     confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
     print("confusion matrix")
     print(confmat)
     print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
     print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
     print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))
   


# In[7]:



list_of_pred_labels = []


for test_x in X_test:
    temp_pred = predict(test_x)   
    list_of_pred_labels.append(   temp_pred[0][0]  )
    

    
print(list_of_pred_labels)
print("true labels below")
print(y_test)


#######################################################

######################################

# y_pred = np.array([  0,0,0,1,1,1,2,2,2,1   ])
# y_test = np.array([  0,0,1,1,1,2,2,2,2,1   ])

######################################

# accuracy_result = accuracy(y_pred, y_test)
# print(    accuracy_result    )


#######################################################

        
#print_stats_percentage_train_test('knn', y_test, y_pred)


# In[8]:


y_pred = np.array(    list_of_pred_labels    )

print(y_pred)
print("true labels below")
print(y_test)


# In[9]:


res = print_stats_percentage_train_test('knn', y_test, y_pred)

print(    res   )


# In[ ]:




