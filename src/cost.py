'''
Definitions of cost functions

Input
-----
h_x - array of hypothesis representation function, 
    (features: x_0=1, x_1=x, x_2=x^2,..., x_n=x^n)
y - array of observed values

'''

import numpy as np



def cost(theta, input_data ,y):
    '''
    cost function taken: square of l2 norm
    
    Input
    -----
    theta - polynomial coefficients
    input_data - values of all features observed
    y     - observation result array

    Output
    ------
    cost - value of cost function
    dcost - array of derivatives (over theta) of cost function
    '''
    h_x = [ np.matmul(theta,x) for x in input_data]

    cost = np.sum((h_x-y)**2)/(2*len(y))
    dcost = [np.sum((h_x-y)*input_data[i])/len(y) for i in len(theta)]

    return (cost, dcost)


def gradient_descent(init_theta, input_data, y, alpha, cost_function=cost):
    '''
    compute gradient descent for minimalisation of cost function
    
    Input
    -----
    init_theta - initial values of optimized parameters
    input_data - data against which theta is optimized
    y          - values of observations
    alpha      - learning rate of gradient descent
    cost_function - function to be optimized

    Output
    ------
    theta  - optimized vectore
    '''
    theta = init_theta
    imax=15

    convergence_condition = False
    i = 0

    while (not (convergence_condition) and i<imax):
        theta_old = theta
        cost, dcost = cost(theta, input_data ,y)
        theta = theta - alpha * dcost
        if np.sum(theta - theta.old) < tol:
            convergence_condition = True    

    return theta
