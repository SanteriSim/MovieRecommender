
"""
Wrapper class for a recommender based on Funk_SVD (or some other method we decide
                                                   to use)

Attributes:
----------
_ratings: pandas.DataFrame with columns ["userId", "movieId", "rating"]
_P: User coefficients in latent space (N_latent factors x N_users)
_Q: Item coefficients in latent space (N_latent_factors x N_items)

"""

class Funk_SVD_recommender:
    
    def __init__(self, ratings, num_LF):
        self._ratings = ratings
        self._num_LF = num_LF # number of latent factors to be used
        
        # Initialize parameters somehow (maybe via private method, as input attributes,
        # or by explicitly calling some initialization method)
        self._P = None
        self._Q = None
        self._user_bias = None
        self._item_bias = None
        
        
    def train(self, learning_rate = 0.05, reg_coefficient = 0.1):
        '''
        Trains the model using stochastic gradient descent
        '''
        return None
          
    def predict_rating(self, user, item):
        '''
        Predicts ratings for given set of users and items
    
        Parameters
        ----------
        user : user ids
        item : item ids

        Returns
        -------
        ratings: pandas.DataFrame with the predicted ratings

        '''
        ratings = None
        return ratings
             
    
    def model_error(self):
        '''
        Calculates e.g. the Frobenius norm between the actual ratings and the current estimate P^T.Q
        '''
        err = None
        return err
        
    def update_rating(self, rating, user, item):
        '''
        Updates or adds a new rating to the rating matrix
        
        Parameters
        ----------
        rating : new rating to be added
        user : user index
        item : item index
        
        '''
        return None
        
    def remove_rating(self, u, i):
        '''
        Parameters
        ----------
        user : user index
        item : item index
        '''
        return None

    # ... and other public methods, feel free to add.
    
    
    # I guess depends on the model if this is needed explicitly
    def _loss(self, r, p, q, reg_coef):
        
        '''
        Implements a loss between a single rating and corresponding estimate p^T.q
        to be used in training

        Parameters
        ----------
        r : ratings(u,i)
        p : P(:,u)
        q : Q(:,i)
        reg_coef : regularization coefficient
        
        Sometimes also user and item biases are considered as parameters to be trained
        
        Returns
        -------
        l :  loss value

        '''
        l = None
        return l
    
    # ... other private methods.