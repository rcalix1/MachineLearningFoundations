#!/usr/bin/env python
# coding: utf-8

# In[1]:



### Singular Value Decomposition for Compression

import numpy as np

X = np.array([[5, 5, 0, 0, 1],
              [4, 5, 1, 1, 0],
              [5, 4, 1, 1, 0],
              [0, 0, 4, 4, 4],
              [0, 0, 5, 5, 5],
              [1, 1, 4, 4, 4]], dtype=np.float32)
             
print(X)

#################################

U, S, V = np.linalg.svd(X, full_matrices=False)

print("\nU=", U, "\n\nS=", S, "\n\nV=", V)


#################################

## calc energy

S_square = np.square(S) 
print(S_square)

energy = np.sum(S_square[:2]) / np.sum(S_square)
print(energy)


#################################

New_X = U[ :, :2]

print(New_X)

#################################


# In[ ]:




