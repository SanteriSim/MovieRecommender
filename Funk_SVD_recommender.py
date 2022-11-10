
"""
Wrapper class for a recommender based on Funk_SVD

Attributes:
----------
_ratings: pandas.DataFrame with columns ["userId", "movieId", "rating"]
_P: User coefficients in latent space (N_latent factors x N_users)
_Q: Item coefficients in latent space (N_latent_factors x N_items)

"""
import numpy as np

class Funk_SVD_recommender:
    
    def __init__(self, ratings, n_latent_factors, n_users, n_items):
        self._ratings = ratings
        self._n_latent_factors = n_latent_factors # number of latent factors to be used
        self._n_users = n_users # Number of users in the dataset (not all are necessarily included in training set)
        self._n_items = n_items # Number of items in the dataset (-..-)
        self._global_avg = ratings["rating"].mean(axis=0)
        
        # Initialize parameters somehow
        # TODO
        self._P = np.zeros(shape = [self._n_latent_factors,  self._n_users])
        self._Q = np.zeros(shape = [self._n_latent_factors,  self._n_items])
        
        # Either zeros or user avg.
        self._user_bias = np.zeros(shape = [self._n_users,])
        self._item_bias = np.zeros(shape = [self._n_items,])
        
        
    def train(self, learning_rate = 0.01, reg_coefficient = 0, tolerance = 1e-4, max_epochs = 5):
        
        '''
        Trains the model using stochastic gradient descent: TODO
        '''
        return None
          
    def predict_rating(self, userId, itemId):
        '''
        Predicts ratings for given set of users and items. 
    
        Parameters
        ----------
        user : user ids
        item : item ids

        Returns
        -------
        ratings: If inputs given as integers, returns the corresponding rating as a number (float)
                 If inputs given as DataFrames/np.arrays: returns a DataFrame in similar form as ._ratings
        '''
        if isinstance(userId, int) and isinstance(itemId, int):
            p = self._P[:,userId]
            q = self._P[:,itemId]
            user_bias = self._user_bias[userId]
            item_bias = self._item_bias[itemId]
            ratings = self._global_avg + user_bias + item_bias + np.dot(p,q)
        else:
            #TODO
            ratings = None
        return ratings
             
    
    def model_error(self):
        '''
        Calculates e.g. the Frobenius norm between the actual ratings and the current estimate P^T.Q
        '''
        err = None
        return err
        
    def update_rating(self, rating, userId, itemId):
        '''
        Updates or adds a new rating to the rating matrix
        
        Parameters
        ----------
        rating : new rating to be added
        user : user index
        item : item index
        
        '''
        return None
        
    def remove_rating(self, userId, itemId):
        '''
        Parameters
        ----------
        user : user index
        item : item index
        '''
        return None

    # ... and other public methods, feel free to add.
    
    
    # I guess depends on the model if this is needed explicitly
    def _loss(self, reg_coef):
        
        '''
        Implements a loss between a single rating and corresponding estimate p^T.q
        to be used in training

        Parameters
        ----------
        reg_coef : regularization coefficient
        
        Returns
        -------
        l :  loss value

        '''
        l = None
        return l
    
        
    def _ratings_by_user(self, userId):
        '''
        Finds all ratings made by user userId

        Returns
        -------
        ratings : DataFrame with columns ["ItemId", "rating"]
        '''
        indices = self._ratings["userId"] == userId
        ratings = self._ratings[indices]
        return ratings
    
    def _ratings_by_item(self, itemId):
        '''
        Finds all ratings made to item itemId

        Returns
        -------
        ratings : DataFrame with columns ["userId", "rating"]
        '''
        indices = self._ratings["itemId"] == itemId
        ratings = self._ratings[indices]
        return ratings
    
    ## TODO if needed
    def _user_avg(self):
        avg = self._global_avg * np.ones(shape = [self._n_users,])
        return avg
    
    def _item_avg(self):
        avg = self._global_avg * np.ones(shape = [self._n_items,])
        return avg
    
    # ... other auxiliary private methods.