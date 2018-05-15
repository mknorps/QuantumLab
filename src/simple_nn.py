import numpy as np


sigmoid = lambda x: 1/(1+np.exp(-x))

class SimpleNN():
    '''
    simple neural network class
    with one hidden layer
    '''

    def __init__(self,n):
        # number of cells in hidden layer
        self.n = n
        # activation matrix
        self.a = np.empty(n) 
        # weights matrix
        self.w = np.empty([n,n+1])

    def _initialize_a(n):

        init_a = np.random.rand(n) 
        return init_a

    def _initialize_w(n):
        init_w = np.random.rand([n+1,n]) 
        return init_w

    def compute_activation(self, sample, act_function=sigmoid):
        '''
        compute activation value of cells in hidden layer

        Input
        -----
        sample - one samples with features (x0,x1,x2,...,xn)

        '''

        self.a = [act_function(np.matmul(self.w[i,:], sample) ) for i in range(n)]


    def train():

        pass

    def test():

