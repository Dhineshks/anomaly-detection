import numpy as np
from numpy import genfromtxt

#genfromtxt is able to take missing data into account
#delimiter uses comma as seperator for parsing the file
def read_dataset(filepath,delimiter=','):
    return genfromtxt(filepath,delimiter=delimiter)


