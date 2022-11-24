'''
Main script for analysis
'''
import pandas as pd
import numpy as np
import pickle
import recommender
#%% Setting up the data

file_name = "ml-latest-small/ratings.csv"

# Read all data
ratings_all = pd.read_csv(file_name,
                       header = 0,
                       usecols = ["userId", "movieId", "rating"])


ratings_all.rename(columns = {'movieId':'itemId'}, inplace = True)

# Map the ids to range (1,N)
userIds = ratings_all["userId"].to_numpy()
itemIds = ratings_all["itemId"].to_numpy()
userIds_unique = np.unique(ratings_all["userId"].to_numpy())
itemIds_unique = np.unique(ratings_all["itemId"].to_numpy())
n_users = len(userIds_unique)
n_items = len(itemIds_unique)

user_update = {}
item_update = {}
for old_id, new_id in zip(userIds_unique, range(n_users)):
    user_update[old_id] = new_id
for old_id, new_id in zip(itemIds_unique, range(n_items)):
    item_update[old_id] = new_id
for i in range(len(userIds)):
    userIds[i] = user_update[userIds[i]] + 1
for i in range(len(itemIds)):
    itemIds[i] = item_update[itemIds[i]] + 1
##
ratings_all["userId"] = userIds
ratings_all["itemId"] = itemIds

# Randomly shuffle the data 
ratings_all = ratings_all.sample(frac = 1)

#Divide into training set, validation set and test set.
n_train = int(np.ceil(0.8*ratings_all.shape[0]))
ratings_train = ratings_all[:n_train]
ratings_tmp = ratings_all[n_train:]
n_validation = int(np.ceil(0.5*ratings_tmp.shape[0]))
ratings_validation = ratings_tmp[:n_validation]
ratings_test = ratings_tmp[n_validation:]

#%% Train the model by using different values of regularizations coefficients and number of latent factors

reg_coefficients = [0.05, 0.1, 0.5, 1.0]
n_latent_factors = [5, 10, 15, 20]

#%% Train and serialize to file

for i in range(len(reg_coefficients)):
    for j in range(len(n_latent_factors)):
        model = recommender.FunkSVD(ratings = ratings_train, n_latent_factors = n_latent_factors[j], n_users = n_users, n_items = n_items)
        model.train(learning_rate = 0.005, reg_coefficient = reg_coefficients[i], n_epochs = 500)
        
        fname = 'trained_models/FunkSVD_{0:d}_{1:d}.pkl'.format(i, j)
        with open(fname, 'wb') as file:
            pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)        

#%% Check the performance for validation set and choose the best hyperparameters

rms_error = np.zeros(shape = [len(reg_coefficients), len(n_latent_factors)])

for i in range(len(reg_coefficients)):
    for j in range(len(n_latent_factors)):
        fname = 'trained_models/FunkSVD_{0:d}_{1:d}.pkl'.format(i, j)
        with open(fname, 'rb') as file:
            model = pickle.load(file)
            ratings_pred = model.predict(ratings_validation["userId"], ratings_validation["itemId"])
            mse = np.linalg.norm(ratings_pred["rating"].to_numpy()-ratings_validation["rating"].to_numpy())**2/n_validation
            rms_error[i,j] = np.sqrt(mse)

min_idx_flat = np.argmin(rms_error)
min_idx = np.unravel_index(min_idx_flat, shape = rms_error.shape)

#%%

reg_coefficient = reg_coefficients[min_idx[0]]
n_lf = n_latent_factors[min_idx[1]]

fname = 'trained_models/FunkSVD_{0:d}_{1:d}.pkl'.format(min_idx[0], min_idx[1])
with open(fname, 'rb') as file:
    model = pickle.load(file)

ratings_test_pred = model.predict(ratings_test["userId"], ratings_test["itemId"])
mse = np.linalg.norm(ratings_test_pred["rating"].to_numpy()-ratings_test["rating"].to_numpy())**2/n_validation
rms_test = np.sqrt(mse)
print(rms_test)