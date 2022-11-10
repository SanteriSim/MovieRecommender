'''
Main script for analysis
'''
import pandas as pd
import numpy as np
from recommender import FunkSVD

#%% Setting up the data

file_name = "ml-latest-small/ratings.csv"

# Read all data
ratings_all = pd.read_csv(file_name,
                       header = 0,
                       usecols = ["userId", "movieId", "rating"])


ratings_all.rename(columns = {'movieId':'itemId'}, inplace = True)
n_users = max(ratings_all["userId"])
n_items = max(ratings_all["itemId"])


# Randomly shuffle the data 
ratings_all = ratings_all.sample(frac = 1)

#Divide into training set and test set. (Perhaps validation set also needed)
n_train = int(np.ceil(0.05*ratings_all.shape[0]))

ratings_train = ratings_all[:n_train]
ratings_test = ratings_all[n_train:]
#%%

model = FunkSVD(ratings = ratings_train, n_latent_factors = 10, n_users = n_users, n_items = n_items)
model.train(learning_rate = 0.01, reg_coefficient = 0.02)
model.L2_error()
