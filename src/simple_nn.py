import numpy as np


sigmoid = lambda x: 1/(1+np.exp(-x))



class SimpleNN():
    '''
    simple neural network class
    with one hidden layer
    '''

    def __init__(self,n,nn, act_function=lambda x :x, init_random=False):
        # order of polynomial (number of nodes in input layer-1)
        self.n = n
        # number of cells in hidden layer
        self.nn = nn
        # activation function for neurons in hidden layer
        self.g = act_function
        # weights matrix
        self.W1 = np.empty([nn,n+1])
        self.W2 = np.empty(nn+1) #+1 for bias node

        if init_random:
            self.W1 =_initialize_W1(self) 
            self.W2 =_initialize_W2(self) 

    def __repr__(self):
        '''
        String representation of SimpleNN object
        '''
        representation = '''
order of polynomial, n: {}
number of cells in hidden layer, nn: {}
weight matrix, W1:
{}
weight matrix, W2:
{}
output, h: {}
        '''.format(self.n,self.nn,self.W1,self.W2,self.h)

        return representation



    def _initialize_W1(self):
        init_w = np.random.rand([self.nn,self.n+1]) 
        return init_w

    def _initialize_W2(self):
        init_w = np.random.rand(self.n+1) 
        return init_w

    def forward_propagate(self,sample):
        '''
        compute fprward propagation of the neural network 
        for one sample vector of input data (x0,x1,...xn)

        Input
        -----
        sample - one samples with features (x0,x1,x2,...,xn)
        act_function - activation function

        '''
        # polynomial - the hypothesis representation
        W1_times_sample = np.matmul(self.W1, sample)
        # hidden layer
        A1 =[ self.g(val) for val in  W1_times_sample]
        # output
        A_with_bias = np.append([1],A1)
        A2 = self.g(np.matmul(self.W2,A_with_bias))
        return A2,A1

    def cost_nn(self,data,y):

        print (data)
        print(data.T)
        y_pred = np.array([ self.forward_propagate(vx) for vx in data.T])

        print(y_pred)

        l = len(y)
        c = np.sum((y_pred-y)**2)/l



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


