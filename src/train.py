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
    feature_data = pre.feature_scaling(raw_data,n)
    data = pre.split_data(feature_data)

    print([(k,len(v)) for k,v in data.items()])


    # create instance of the neural network
    #-------------------------------------------------
    nn = n*2
    model = ionn.InOutNN(n, nn)
    print(model)


    # train the model 
    #-------------------------------------------------
    poly_coeffs = model.train(data['train'][:,:-1], data['train'][:,-1])

    print(poly_coeffs)
    print(np.polyfit(data['train'][:,1], data['train'][:,-1], deg=n, full = True))
    

    # test the model 
    #-------------------------------------------------
    model.test()


    # dump the model - Python object serialisation
    #-------------------------------------------------
    model_file = "MODEL/simple_nn_n{}.pcl".format(n)
    with open(model_file,'wb') as f:
        pickle.dump(model, f)


    return poly_coeffs
