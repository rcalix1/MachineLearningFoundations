#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

##############################################################

##  Reading dataset (MovieLens 1M movie ratings dataset: 
## downloaded from https://grouplens.org/datasets/movielens/1m/)

#############################################################

data = pd.read_csv('recommender/ratings.csv')

users  = data['userId'].unique()   ## list all users
movies = data['movieId'].unique() ## list all movies

print("number of users ", len(users))
print("number of movies", len(movies))

print(data.head())

##################################################################

data_ratings = pd.read_csv('recommender/ratings.csv')

data_movies = pd.read_csv('recommender/movies.csv')

#################################################################

##train = data.sample(frac=0.8, random_state=200)
##test =  data.drop(train.index)

################################################################
#Creating the rating matrix (rows as movies, columns as users)


ratings_mat = np.zeros( shape=(9724,610),  dtype=np.uint8  )
print(ratings_mat.shape)

##################################################################


ratings_array  = data_ratings.rating.values
movieIds_array = data_ratings.movieId.values
userIds_array  = data_ratings.userId.values

###################################################################


movies_uniques_dict_ref_real = {}
movies_uniques_dict_real_ref = {}
movies_uniques  = np.unique(movieIds_array)
for idx, res in enumerate(movies_uniques):
    movies_uniques_dict_ref_real[idx]=res 
    movies_uniques_dict_real_ref[res]=idx 

users_uniques_dict_ref_real = {}
users_uniques_dict_real_ref = {}
users_uniques   = np.unique(userIds_array)
for idx, res in enumerate(users_uniques):
    users_uniques_dict_ref_real[idx]=res  
    users_uniques_dict_real_ref[res]=idx
    
########################################################################

#print(movies_uniques_dict)

print(len(ratings_array))
#print(movies_uniques)
#print(users_uniques)


for i in range(   len(ratings_array)    ):
    ref_u = users_uniques_dict_real_ref[   userIds_array[i]   ]
    ref_m = movies_uniques_dict_real_ref[    movieIds_array[i]   ]
    ratings_mat[ref_m, ref_u] = ratings_array[i]

##################################################################

#Normalizing the matrix(subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T


##################################################################


#Computing the Singular Value Decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)

print(A.shape)


##################################################################

U, S, V = np.linalg.svd(A)

print(U.shape)
print(S.shape)
print(V.shape)

#print(U)
#print(S)
#print(V)

#################################################################

'''

#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]
    
#####################################################################

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])
        
#########################################################################

#k-principal components to represent movies, movie_id to find recommendations, top_n print n results   

k = 50
movie_id = 10 # (getting an id from movies.dat)
top_n = 10


sliced = V.T[:, :k] # representative data


indexes = top_cosine_similarity(sliced, movie_id, top_n)

#######################################################################

#Printing the top N similar movies
print_similar_movies(data_movies, movie_id, indexes)

'''

print("<<<<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>>>")


# In[ ]:




