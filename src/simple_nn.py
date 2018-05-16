import numpy as np


sigmoid = lambda x: 1/(1+np.exp(-x))

class SimpleNN():
    '''
    simple neural network class
    with one hidden layer
    '''

    def __init__(self,n,nn, init_random=False):
        # order of polynomial (number of nodes in input layer-1)
        self.n = n
        # number of cells in hidden layer
        self.nn = nn
        # activation matrix - hidden layer
        self.A = np.empty(nn) 
        # weights matrix
        self.W1 = np.empty([nn,n+1])
        self.W2 = np.empty(nn+1) #+1 for bias node

        if init_random:
            self.A = _initialize_A(self) 
            self.W1 =_initialize_W1(self) 
            self.W2 =_initialize_W2(self) 

        # output
        self.h = 0

    def __repr__(self):
        '''
        String representation of SimpleNN object
        '''
        representation = '''
order of polynomial, n: {}
number of cells in hidden layer, nn: {}
activation matrix, A:
{}
weight matrix, W1:
{}
weight matrix, W2:
{}
output, h: {}
        '''.format(self.n,self.nn,self.A,self.W1,self.W2,self.h)

        return representation



    def _initialize_A(self):
        init_a = np.random.rand(self.n) 
        return init_a

    def _initialize_W1(self):
        init_w = np.random.rand([self.nn,self.n+1]) 
        return init_w

    def _initialize_W2(self):
        init_w = np.random.rand(self.n+1) 
        return init_w

    def forward_propagate(self,sample,act_function=sigmoid):
        '''
        compute activation value of cells in hidden layer
        for one sample vector of input data (x0,x1,...xn)

        Input
        -----
        sample - one samples with features (x0,x1,x2,...,xn)
        act_function - activation function

        '''
        # polynomial
        W1_times_sample = np.matmul(self.W1, sample)
        # hidden layer
        self.A =[ act_function(val) for val in  W1_times_sample]
        # output
        A_with_bias = np.append([1],self.A)
        self.h = act_function(np.matmul(self.W2,A_with_bias))

    def back_propagate(self):
        pass


    def compute_poly_coeffs(self):
        '''
        compute resulting polynomial coefficients
        '''
        coeffs = []
        return coeffs


    def train(self):
        pass

    def test(self):
        pass


