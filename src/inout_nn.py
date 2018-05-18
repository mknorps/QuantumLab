import numpy as np

class InOutNN():
    '''
    simple neural network class
    with no hidden layer
    activation function: identity
    '''

    def __init__(self,n,init_random=False):
        # order of polynomial (number of nodes in input layer-1)
        self.n = n
        # weights matrix
        self.W1 = np.empty(n+1)

        # scaling
        self.scaling = np.empty(n+1)

        if init_random:
            self.W1 =self._initialize_W1() 


    def _initialize_W1(self):
        init_w = np.random.rand(self.n+1) 
        return init_w

    def __repr__(self):
        '''
        String representation of InOutNN object
        '''
        representation = '''
order of polynomial, n: {}
weight matrix, W1:
{}
        '''.format(self.n,self.W1)
        return representation

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
        return W1_times_sample

    def cost(self,input_data ,y, lam=0.0):
        '''
        cost function taken: square of l2 norm + regularisation
        
        Input
        -----
        input_data - values of all features observed
        y     - observation result array

        Output
        ------
        cost - value of cost function
        dcost - array of derivatives (over self.W1) of cost function
        '''
        h_x = np.array([ np.matmul(self.W1,x) for x in input_data])
        l = len(y)

        cost = np.sum((h_x-y)**2)/(2*l) + lam*np.sum(self.W1**2)/(2*l)
        dcost = np.array([np.sum( np.multiply((h_x-y),input_data[:,i]) )/l 
            + lam*self.W1[i]/l for i in range(self.n+1)])

        return (cost, dcost)

    def gradient_descent(self, input_data, y, alpha=1, tol=0.0001, itmax=100, verbose=True):
        '''
        compute gradient descent for minimalisation of self.cost function
        
        Input
        -----
        input_data - data against which theta is optimized
        y          - values of observations
        alpha      - learning rate of gradient descent
        tol        - tolerance: the boundary for the difference 
                     between cost function in two consecutive iterations 
        itmax      - maximum number of iterations

        Output
        ------
        self.W1  - optimized vectore
        '''
        theta = self.W1

        # regularisation parameter
        #lam = 0.01*len(y)/alpha
        lam=0
     
        convergence_condition = False
        i = 0
        cost_prev =0 

        while (not (convergence_condition) and i<itmax):
            i = i+1
            theta_old = self.W1
            cost, dcost = self.cost(input_data ,y, lam=lam)
            if verbose:
                print(i,"cost: ",cost, self.W1, dcost)
            self.W1= self.W1 - alpha * dcost
            # jumping from the wrong minimum
            #if np.abs(cost_prev-cost)<tol:
            #    self.W1 = self.W1*(0.01*np.random.rand(1)+1)

            cost_prev = cost
            if cost< tol:
                convergence_condition = True    



    def back_propagate(self):
        # gradient descent is enough
        pass

    def polynomial_coefficients(self, means, stds):


        theta_new = np.divide(self.W1[1:], stds[1:])
        free_term = - np.sum(theta_new * means[1:])
        scaling = np.insert(theta_new, 0,free_term)
        
        self.scaling = scaling
        poly_coeffs = np.insert(scaling[1:], 0, self.W1[0]+free_term)

        return poly_coeffs

    def train(self, data, y, alpha=1.01, tol=0.0001, itmax=400, verbose=True):

        self.gradient_descent(data,y, alpha, tol, itmax, verbose)

        return np.flip(self.W1, 0)

    def test(self):
        pass


