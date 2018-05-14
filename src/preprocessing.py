import numpy as np
import os
import csv


def read_data():
    '''
    read dataset from CSV to numpy array
    '''
    dir_path = os.getcwd()
    points = np.genfromtxt(dir_path+ "/../DATA/marie-knorps.csv", delimiter=',')

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

    return normalised



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
    '''
    m = len(data)
    features = [] 
    x_rescaled = _normalise_array(data[:,0])
    y_rescaled = _normalise_array(data[:,1])

    for i in range(n):
        features.append([x**(i+1) for x in x_rescaled])

    features.append(y_rescaled)
    features = np.array(features).reshape((m,n+1))

    return features
