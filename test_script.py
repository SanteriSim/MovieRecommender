'''
Main script for analysis
'''
import pandas as pd
import numpy as np
import Funk_SVD_recommender


#%% Setting up the data

file_name = "ml-latest-small/ratings.csv"

# Read all data
ratings_all = pd.read_csv(file_name,
                       header = 0,
                       usecols = ["userId", "movieId", "rating"])

# Randomly shuffle the data 
ratings_all = ratings_all.sample(frac = 1)

#Divide into training set and test set. (Perhaps validation set also needed)
N_train = int(np.ceil(0.8*ratings_all.shape[0]))

ratings_train = ratings_all[:N_train]
ratings_test = ratings_all[N_train:]

#%% Training the model, etc.

model = Funk_SVD_recommender(ratings = ratings_train, num_LF = 20)

model.train(learning_rate = 0.05, reg_coefficient = 0.1)