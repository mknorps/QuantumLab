import numpy as np


class InOutNN():
    '''
    simple neural network class
    with no hidden layer
    '''

    def __init__(self,n,nn, init_random=False):
        # order of polynomial (number of nodes in input layer-1)
        self.n = n
        # weights matrix
        self.W1 = np.empty(n+1)

        if init_random:
            self.W1 =_initialize_W1(self) 

        # output
        self.h = 0


    def _initialize_W1(self):
        init_w = np.random.rand(self.n+1) 
        return init_w

    def forward_propagate(self,sample):
        '''
        compute output
        for one sample vector of input data (x0,x1,...xn)

        Input
        -----
        sample - one samples with features (x0,x1,x2,...,xn)
        act_function - activation function

        '''
        # polynomial
        W1_times_sample = np.matmul(self.W1, sample)
        self.h = W1_times_sample

    def back_propagate(self):

        pass

    def train():
        pass

    def test():
        pass


