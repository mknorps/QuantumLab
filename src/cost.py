'''
Definitions of cost functions

Input
-----
h_x - array of hypothesis representation function, 
    (features: x_0=1, x_1=x, x_2=x^2,..., x_n=x^n)
y - array of observed values

'''

import numpy as np




def J_l2_norm(h_x,y):
    '''
    l2 norm
    '''
    assert (len(h_x)==len(y))

    cost = np.sum((h_x-y)**2)/len(y)

    return cost


