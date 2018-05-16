
import pickle

def estimate(x):

    print("Hello, you are in estimate module")
    
    # pick up the model from file
    #-------------------------------------------------
    model_file = "MODEL/simple_nn_n2.pcl"
    with open(model_file,'rb') as f:
        simple_nn = pickle.load(f)


    print(simple_nn)
