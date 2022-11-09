
"""
Wrapper class for a recommender based on Funk_SVD (or some other method we decide
                                                   to use)

Attributes:
----------
_ratings: Rating matrix (N_users x N_items) in a some reasonable data structure
_P: User coefficients in latent space (N_latent factors x N_users)
_Q: Item coefficients in latent space (N_latent_factors x N_items)

"""

class Funk_SVD_recommender:
    
    def __init__(self, ratings, num_LF):
        self._ratings = ratings
        self._num_LF = num_LF # number of latent factors to be used
        
        # initialize parameters somehow
        self._P = None
        self._Q = None
        self._user_bias = None
        self._item.bias = None
        
        
    def train(self, learning_rate = 0.05, reg_coefficient = 0.1):
        
        '''
        Implements the training of the model using stochastic gradient descent
        '''
        return None
          
    def model_error(self):
        '''
        Calculares e.g. the Frobenius norm between the actual ratings and the current estimate P^T.Q
        '''
        return None
        
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

    # I guess depends on the model if this is needed explicitly
    def _loss(self, r, p, q, reg_coef):
        
        '''
        Implements a loss between a single rating and corresponding estimate p^T.q
        to be used in training

        Parameters
        ----------
        r : ratings(u,i)
        p : P(:,i)
        q : Q(:,I)
        reg_coef : regularization coefficient
        
        Sometimes also user and item biases are considered as parameters to be trained
        
        Returns
        -------
        l : 

        '''
        return None
    
    