import glob
import os
import pickle
import numpy as np

def estimate(x):
    '''
    estimate value of recently created network for input value x
    '''
    # get the latest file in a current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path+'/../MODEL/*.pcl')
    latest_file = max(list_of_files, key=os.path.getctime)

    # pick up the model from file
    #-------------------------------------------------
    model_file = latest_file 
    with open(model_file,'rb') as f:
        inout_nn = pickle.load(f)


    input_data = np.array([x**i for i in range(inout_nn.n+1)])
    out = inout_nn.forward_propagate_rescaled(input_data) 

    print(out)
