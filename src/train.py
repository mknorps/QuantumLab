import numpy as np
from src import preprocessing as pre
from src import inout_nn as ionn
import pickle 
import os

def train(n, path_to_csv):
    '''
    train the model

    Input
    -----
    n  - polynomial degree
    path_to_csv - training set path
    '''
    # prepare data
    #-------------------------------------------------
    # load data
    raw_data = pre.read_data(path_to_csv)
    # rescale data
    feature_data, means,stds = pre.feature_scaling(raw_data,n)
    # split to train and test sets
    data = pre.split_data(feature_data)


    # create instance of the neural network
    #-------------------------------------------------
    model = ionn.InOutNN(n, init_random=True)


    # train the model 
    #-------------------------------------------------
    model.train(data['train'][:,:-1], data['train'][:,-1],
            alpha=0.5, itmax=20000, verbose = False)
    #compute polynomial coefficients (from rescaled features)
    poly_coeffs = model.polynomial_coefficients(means, stds)

    print(np.flip(poly_coeffs, axis=0))
    

    # test the model 
    #-------------------------------------------------
    train_err, test_err = model.test(data['train'][:,:-1], data['train'][:,-1],
            data['test'][:,:-1], data['test'][:,-1])


    # dump the model - Python object serialisation
    #-------------------------------------------------
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = dir_path+"/../MODEL/inoutnn_{}.pcl".format(n)
    with open(model_file,'wb') as f:
        pickle.dump(model, f)


    return poly_coeffs
