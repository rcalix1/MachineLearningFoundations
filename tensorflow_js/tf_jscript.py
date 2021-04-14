#!/usr/bin/env python
# coding: utf-8

# In[1]:

#########################################

## >>pip install tensorflowjs
## >>tensorflowjs_converter --input_format=keras modelRC.h5 model_js

#########################################


import tensorflow as tf
import numpy as np

#########################################

x = []
y = []

for _ in range(10000):
    num = np.random.randint(0, 101)
    x.append(num)
    y.append(num % 2)

#########################################

#print(x)
#print(y)

#########################################

model = tf.keras.Sequential()
model.add(    tf.keras.layers.Dense(input_shape=(1,), units=64, activation='sigmoid')      )
model.add(    tf.keras.layers.Dense(units=32, activation='sigmoid')     )
model.add(    tf.keras.layers.Dense(units=16, activation='sigmoid')     )
model.add(    tf.keras.layers.Dense(units=8,  activation='sigmoid')     )
model.add(    tf.keras.layers.Dense(units=1,  activation='sigmoid')     )

model.compile('adam', 'binary_crossentropy')

model.fit(x, y, epochs=100)

model.save('modelRC.h5')

#########################################


# In[ ]:




