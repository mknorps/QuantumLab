import numpy as np
import os
import csv


def read_data(path_to_csv):
    '''
    read dataset from CSV to numpy array
    '''
    # TODO userproof
    points = np.genfromtxt(path_to_csv, delimiter=',')

    return points


def _normalise_array(a):
    '''
    normalise array by substracting mean and dividing by standard deviation
    '''
    mean = np.mean(a)
    std = np.std(a)

    normalised = (a-mean)

    if std>0.0:
        normalised = normalised/std

    return (normalised, mean, std)


def split_data(data,train=0.8, test=0.2):
    '''
    randomly split data into training, test and validation sets

    Input
    -----
    data  - array of data of shape (m,2), where m is a number of samples
    train - fraction of data that will be used for training
    test  - fraction of data that will be used for evaluating the model

    Output
    ------
    splited_data - dictionary of splited data
    '''
    np.random.seed(42) #for reproducability
    np.random.shuffle(data)
    l = len(data)
    splited_data = {"train":[], 'test':[], 'valid':[]}

    if l == 0:
        return splited_data

    l_train = int(train*l)
    l_test = int(test*l)

    assert (l_train+l_test <= l)
    # the remamining data points are used for validation
    l_valid = l - l_train - l_test


    splited_data['train'] = data[:l_train,:] 
    splited_data['test']  = data[l_train:l_train+l_test,:] 
    splited_data['valid']  = data[l_train+l_test:,:] 

    return splited_data


def feature_scaling(data,n):
    '''
    scale input features for the model
    create new set of rescaled features based on the order of target polynomial 

    Input
    -----
    data - array of points (shape (m,2)) to be rescaled
    n - order of polynomial

    Output
    ------
    features - array of normalised features
    scales   - array of scaling values
    '''
    m = len(data)
    features = [np.ones(m)] #x0
    means = [1]
    stds = [0]
    if n>0:
        norm, mean, std = _normalise_array(data[:,0])
        features.append(norm) #x1
        means.append(mean)
        stds.append(std)
        for i in range(n-1):
            # take last feature and rescale it
            x_i_plus_1 = [x**(i+2) for x in data[:,0]]
            norm, mean, std = _normalise_array(x_i_plus_1)
            features.append( norm) 
            means.append(mean)
            stds.append(std)
        
    features.append(data[:,1])    
    features = np.array(features).T
    means = np.array(means)
    stds = np.array(stds)

    return (features,means,stds)
