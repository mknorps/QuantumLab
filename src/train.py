import numpy as np
from src import preprocessing as pre
from src import inout_nn as ionn
import pickle 

def train(n, path_to_csv):
    '''
    train the model

    Input
    -----
    n  - polynomial degree
    '''

    print("Hello, you are in train module")

    # prepare data
    #-------------------------------------------------
    raw_data = pre.read_data(path_to_csv)
    feature_data, means,stds = pre.feature_scaling(raw_data,n)
    data = pre.split_data(feature_data)


    # create instance of the neural network
    #-------------------------------------------------
    nn = n*2
    model = ionn.InOutNN(n, init_random=True)
    model.polynomial_coefficients(means, stds)


    # train the model 
    #-------------------------------------------------
    model.train(data['train'][:,:-1], data['train'][:,-1], itmax=1000, verbose=False)
    poly_coeffs = model.polynomial_coefficients(means, stds)

    print("model: ", np.flip(poly_coeffs, axis=0))
    print("polyfit: ",np.polyfit(raw_data[:,0], raw_data[:,1], deg=n))
    

    # test the model 
    #-------------------------------------------------
    model.test()


    # dump the model - Python object serialisation
    #-------------------------------------------------
    model_file = "MODEL/simple_nn_n{}.pcl".format(n)
    with open(model_file,'wb') as f:
        pickle.dump(model, f)


    return poly_coeffs
