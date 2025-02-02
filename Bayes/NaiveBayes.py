#!/usr/bin/env python
# coding: utf-8

# In[1]:



## Naive Bayes 
## 2021

############################################

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

############################################


def train(X_train, y_train): 
    n_samples, n_features = X_train.shape       ## 120 samples x 4 features, if using iris
    list_of_classes = np.unique(y_train)  ## for iris there are 3 classes so you get [0, 1] 
    n_classes = len(list_of_classes)      ##  2 classes

    ## calculate mean, variance for the gaussian, and the priori probability for the classes
    means = np.zeros( (n_classes, n_features), dtype=np.float64      )
    variances = np.zeros( (n_classes, n_features), dtype=np.float64      )
    priori = np.zeros( n_classes, dtype=np.float64      )
   
    for idx, c in enumerate(list_of_classes):
        X_c = X_train[y_train==c]
        means[idx, : ] = X_c.mean(axis=0)
        variances[idx, :] = X_c.var(axis=0)
        priori[idx] = X_c.shape[0] / float(n_samples)
   
    return list_of_classes, means, variances, priori
    
    
###################################################################################

def predict_set(X_test, list_of_classes, means, variances, priori):
    y_pred_vector = [predict(x, list_of_classes, means, variances, priori) for x in X_test]
    return np.array(  y_pred_vector   )
    
    
###################################################################################

def predict(x_one_sample, list_of_classes, means, variances, priori):
    result_per_class = []
    
    ## calculate the posterior probabilities for each class and select the most 
    ## likely class
    for idx, c in enumerate(list_of_classes):
        prior = np.log(   priori[idx]   )
        posterior = gaussian(idx, x_one_sample, means, variances)  ## return vector of 4
        posterior = np.sum(   np.log(  posterior ) )
        prob_per_class = posterior + prior
        result_per_class.append(   prob_per_class   )
    
    ## return class with highest probability 
    return list_of_classes[     np.argmax(   np.array(result_per_class)   )     ]
        
    
###################################################################################



## pdf = probability density function
    
def gaussian(idx, xs, means, variances):
    numerator =  np.exp(  ( -(xs - means[idx])**2 ) / (2 * variances[idx])    )
    denominator =  np.sqrt(     2 * np.pi * variances[idx]      )
    return numerator / denominator
    
    


##################################################################################

X, y = datasets.make_classification(   n_samples = 150, n_features=4, n_classes=2, random_state=123 )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

list_of_classes, means, variances, priori = train(X_train, y_train)

y_pred = predict_set(X_test, list_of_classes, means, variances, priori) 

print(y_pred)
print(y_test)

## print(   accuracy(y_test, y_pred)   )

##################################################################################


# In[ ]:





# In[ ]:





# In[ ]:




