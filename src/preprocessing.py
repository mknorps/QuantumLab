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



