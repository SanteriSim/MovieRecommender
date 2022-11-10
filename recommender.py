
import numpy as np
import pandas as pd

class FunkSVD:
    """
    Wrapper class for a recommender based on Funk SVD

    Attributes:
    ----------

    _P: User coefficients in latent space (N_latent factors x N_users)
    _Q: Item coefficients in latent space (N_latent_factors x N_items)

    """
    def __init__(self, ratings, n_latent_factors, n_users, n_items):
        
        # Initialize auxiliary attributes
        self._n_latent_factors = n_latent_factors # number of latent factors to be used
        self._n_users = n_users # Number of users in the dataset (not all are necessarily included in training set)
        self._n_items = n_items # Number of items in the dataset (-..-)
        self._n_instances = ratings.shape[0] # Number of training instances
        
        # Vectorize dataframe for faster iteration and computation 
        self._ratings = ratings["rating"].to_numpy()
        self._users = ratings["userId"].to_numpy(dtype = int)
        self._items = ratings["itemId"].to_numpy(dtype = int)
        
        # Global average that normalizes the data
        # Note: no explicit normalization is needed because the model takes care of it
        self._global_avg = self._ratings.mean(axis=0)
        
        # Initialize parameters somehow
        
        # TODO: these cannot be zeros
        self._P = np.random.rand(self._n_latent_factors,  self._n_users)-0.5
        self._Q = np.random.rand(self._n_latent_factors,  self._n_items)-0.5
        
        # These can be either zeros or something else
        self._user_bias = np.zeros(shape = [self._n_users,])
        self._item_bias = np.zeros(shape = [self._n_items,])
        
        
    def train(self, learning_rate = 0.005, reg_coefficient = 0.02, tolerance = 1e-5, max_epochs = 100):
        
        '''
        Trains the model using stochastic gradient descent.
        TODO, CURRENTLY TRAINS ONLY FOR ONE EPOCH
        
        '''
        self._train_epoch(learning_rate, reg_coefficient)
        
        # Give some user feedback 
        print(self.L2_error())
        return None
          
    def predict(self, userId, itemId):
        '''
        Predicts ratings for given set of users and items. 
    
        Parameters
        ----------
        user : user ids
        item : item ids

        Returns
        -------
        ratings: If inputs given as integers, returns the corresponding rating as a number (float)
                 If inputs given as np.arrays of same shape, returns an np.array with rating for each pair
                 If inputs given as pd.Series returns a pd.DataFrame in similar form as input data
        '''
        # Input type <int>
        if isinstance(userId, int) and isinstance(itemId, int):
            p = self.user_latent_factors(userId)
            q = self.item_latent_factors(itemId)
            user_bias = self.user_bias(userId)
            item_bias = self.item_bias(itemId)
            ratings = self._global_avg + user_bias + item_bias + np.dot(p,q)
            
        # Input type <numpy.ndarray> or <pandas.Series>
        else:
            # Determine type and copy to numpy variables
            if isinstance(userId, pd.Series) and isinstance(userId, pd.Series):
                users = userId.toNumpy(dtype = int)
                items = itemId.toNumpy(dtype = int)             
            elif isinstance(userId, np.ndarray) and isinstance(userId, np.ndarray):
                users = userId
                items = itemId
            else:
                raise Exception("Invalid input type")
                
            # Check that the arrays are of the same size
            if userId.shape != itemId.shape:
                raise Exception("User and item arrays should be the same size")    
            ratings_tmp = np.zeros(shape = userId.shape)
            
            # Compute the predictions
            for index in range(len(ratings)):
                p = self.user_latent_factors(users[index])
                q = self.item_latent_factors(items[index])
                user_bias = self.user_bias(users[index])
                item_bias = self.item_bias(items[index])
                ratings_tmp[index] = self._global_avg + user_bias + item_bias + np.dot(p,q)
                
            # Determine return type            
            if isinstance(userId, np.ndarray) and isinstance(userId, np.ndarray):
                ratings = ratings_tmp
            elif isinstance(userId, pd.Series):
                ratings = pd.DataFrame()
                ratings["userId"] = users.toList()
                ratings["itemId"] = items.toList()
                ratings["rating"] = ratings_tmp.toList()
        return ratings
             
    
    def L2_error(self):
        '''
        Calculates the L2 norm between the actual ratings and the current prediction
        '''
        dif = self._ratings - self.predict(self._users, self._items)
        err = np.linalg.norm(dif)
        
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

# Use these to access parameters values in order to avoid unnecessary index errors.
# Only use indexing when modifying the parameter values.
    
    def user_latent_factors(self, userId):
        return self._P[:,userId-1]
        
    def item_latent_factors(self, itemId):
        return self._Q[:,itemId-1]
    
    def user_bias(self, userId):
        return self._user_bias[userId-1]
    
    def item_bias(self, itemId):
        return self._item_bias[itemId-1]
    
    # ... and other public methods, feel free to add.
    

    
    def _train_epoch(self, learning_rate, reg_coefficient):
        '''
        Trains the model with SGD by traversing through data once in randomized order.

        Parameters
        ----------
        learning_rate : learning rate
        reg_coefficient : regularization coefficient

        Returns
        -------
        None.

        '''
        indices = list(range(self._n_instances))
        np.random.shuffle(indices)
        
        for index in indices:

            userId = int(self._users[index])
            itemId = int(self._items[index])
            rating = self._ratings[index]
            
            # Current parameters corresponding to the training instance
            p = self.user_latent_factors(userId)
            q = self.item_latent_factors(itemId)
            user_bias = self.user_bias(userId)
            item_bias = self.item_bias(itemId)
            
            # Difference between predicted and real ratings
            dif = rating - self.predict(userId, itemId)
            
            # Update parameters by Funk SGD rule
            self._user_bias[userId-1] = user_bias + learning_rate*(dif - reg_coefficient*user_bias)
            self._item_bias[itemId-1] = item_bias + learning_rate*(dif - reg_coefficient*item_bias)
            self._P[:,userId-1] = p + learning_rate*(dif*q - reg_coefficient*p)
            self._Q[:,itemId-1] = q + learning_rate*(dif*p - reg_coefficient*q)
        
        return None
    
    # ... other auxiliary private methods.