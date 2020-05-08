import numpy as np
from numpy import genfromtxt

#genfromtxt is able to take missing data into account
#delimiter uses comma as seperator for parsing the file
def read_dataset(filepath,delimiter=','):
    return genfromtxt(filepath,delimiter=delimiter)

def estimateGaussian(x):
    m,n = x.shape
    
    #For returning mean and variance of both the columns
    mu = np.zeros(n)
    sigma_square = np.zeros(n)
    
    #mean
    mu = (1/m) * np.sum(x,axis = 0)
    #variance
    sigma_square = (1/m) * np.sum((x - mu) ** 2, axis=0) 
    
    return mu , sigma_square