
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
        self._users = ratings["userId"].to_numpy(dtype = np.int32)
        self._items = ratings["itemId"].to_numpy(dtype = np.int32)
        
        # Global average that normalizes the data
        # Note: no explicit normalization is needed because the model takes care of it
        self._global_avg = self._ratings.mean(axis=0)
        
        # Initialize parameters somehow
        
        # This is pretty arbitrary but seems to work well
        self._P = (np.random.rand(n_latent_factors,  n_users)-0.5)/np.sqrt(n_latent_factors)
        self._Q = (np.random.rand(n_latent_factors,  n_items)-0.5)/np.sqrt(n_latent_factors)
        
        # These can be either zeros or something else
        self._user_bias = np.zeros(shape = [n_users,])
        self._item_bias = np.zeros(shape = [n_items,])
        
        
    def train(self, learning_rate = 0.005, reg_coefficient = 0.02, n_epochs = 100):
        '''
        Trains the model using stochastic gradient descent.
        
        Parameters
        ----------
        learning_rate : learning rate to be used in training
        reg_coefficient : regularization coefficient

        Returns
        -------
        loss : values for loss after each epoch as numpy.array
        '''
        # Testing whether all works as expected
        loss = np.zeros(shape = [n_epochs,])
        for epoch in range(n_epochs):
            self._train_epoch(learning_rate, reg_coefficient)
            loss[epoch] = self._loss(reg_coefficient)
            err = self.RMSE()
            print("Epoch: {0:d}, Loss: {1:.3f}, RMSE: {2:.3f}".format(epoch+1, loss[epoch], err))
        
        return loss
          
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
        # Input type <int> or <np.int32>
        if (isinstance(userId, int) and isinstance(itemId, int)) or \
            (isinstance(userId, np.int32) and isinstance(itemId, np.int32)):           
            p = self.user_latent_factors(userId)
            q = self.item_latent_factors(itemId)
            user_bias = self.user_bias(userId)
            item_bias = self.item_bias(itemId)
            ratings = self._global_avg + user_bias + item_bias + np.dot(p,q)
            
        # Input type <numpy.ndarray> or <pandas.Series>
        else:
            # Determine type and copy to numpy variables
            if isinstance(userId, pd.Series) and isinstance(userId, pd.Series):
                users = userId.to_numpy(dtype = np.int32)
                items = itemId.to_numpy(dtype = np.int32)             
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
            for index in range(len(ratings_tmp)):
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
                ratings["userId"] = pd.Series(users)
                ratings["itemId"] = pd.Series(items)
                ratings["rating"] = pd.Series(ratings_tmp)
        return ratings
    
    def RMSE(self):
        '''
        Calculates the root-mean-squared error of the predicted ratings
        '''
        dif = self._ratings - self.predict(self._users, self._items)
        rmse = np.sqrt(np.sum(dif**2)/self._n_instances)
        
        return rmse
        
    def parameters(self):
        '''
        Returns the model parameters as a tuple.
        
        Returns
        -------
        User latent factor matrix as numpy.array,
        Item latent factor matrix as numpy.array,
        User biases as numpy.array,
        Item biases as numpy.array

        '''
        return self._P, self._Q, self._user_bias, self._item_bias
    
    def initialize_parameters(self, P, Q, user_bias, item_bias, ratings = None, users = None, items = None):
        '''
        Initializes parameters with user input. 

        Parameters
        ----------
        P : User latent factor matrix as numpy.array 
        Q : Item latent factor matrix as numpy.array 
        user_bias : User biases as numpy.array 
        item_bias : Item biases as numpy.array 
        
        Optional: (Use these if there is a risk that the order of users and items
                   in the parameters has changed from what the model was initialized with.)
        ratings: corresponding ratings as numpy.array  or pandas.Series
        users: corresponding userIds as numpy.array or pandas.Series
        items: corresponding itemIds as numpy.array or pandas.Series
        
        Returns
        -------
        None.

        '''
        
        self._P = P; self._Q = Q; 
        self._user_bias = user_bias; self._item_bias = item_bias
        if ratings is None:
            ratings_tmp = self._ratings
        else:
            if isinstance(ratings, pd.Series):
                ratings_tmp = ratings.to_numpy()
            else:
                ratings_tmp = ratings
        if users is None:
            users_tmp = self._users
        else:
            if isinstance(users, pd.Series):
                users_tmp = users.to_numpy()
            else:
                users_tmp = users
        if items is None:
            items_tmp = self._items
        else:
            if isinstance(items, pd.Series):
                items_tmp = items.to_numpy()
            else:
                items_tmp = users
        self._ratings = ratings_tmp
        self._users = users_tmp
        self._items = items_tmp
        return None
    
    def users(self):
        return self._users
    
    def items(self):
        return self._items
    
    def ratings(self):
        return self._ratings

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

            userId = self._users[index]
            itemId = self._items[index]
            rating = self._ratings[index]
            
            # Current parameters corresponding to the training instance
            p = self.user_latent_factors(userId)
            q = self.item_latent_factors(itemId)
            user_bias = self.user_bias(userId)
            item_bias = self.item_bias(itemId)
            
            # Difference between predicted and real ratings
            dif = rating - self.predict(userId, itemId)
            
            # Update parameters:
            self._user_bias[userId-1] = user_bias + learning_rate*(dif - reg_coefficient*user_bias)
            self._item_bias[itemId-1] = item_bias + learning_rate*(dif - reg_coefficient*item_bias)
            self._P[:,userId-1] = p + learning_rate*(dif*q - reg_coefficient*p)
            self._Q[:,itemId-1] = q + learning_rate*(dif*p - reg_coefficient*q)
        
        return None
    
    def _loss(self, reg_coefficient):
        dif = self._ratings - self.predict(self._users, self._items)
        err = np.sum(dif**2)
        reg_bias = np.sum(self._user_bias**2) + np.sum(self._item_bias**2)
        reg_latent_factors = np.sum(self._P**2) + np.sum(self._Q**2)
        loss = err + reg_coefficient*(reg_bias + reg_latent_factors)
        return loss
            
        
        
    # ... other auxiliary private methods.