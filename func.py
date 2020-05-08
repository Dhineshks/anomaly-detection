import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

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

def GaussianDistribution(x,mu,sigma_square):
    l = mu.size
    
    #converting sigma_square into diagonal matrix for determinant calculation
    if sigma_square.ndim == 1:
        sigma_square = np.diag(sigma_square)
        
    x = x - mu 
    
    #gaussian distribution formula
    p = (2 * np.pi) ** (-l / 2) * np.linalg.det(sigma_square) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(x, np.linalg.pinv(sigma_square)) *x ,axis=1))
    
    return p

def visualize(x,mu,sigma_square):
    x1,x2 = np.meshgrid(np.arange(0,35.5,0.5),np.arange(0,35.5,0.5))
    z = GaussianDistribution(np.stack([x1.ravel(),x2.ravel()],axis=1),mu,sigma_square)
    z = z.reshape(x1.shape)
    
    plt.plot(x[:,0],x[:,1],'bx',mec='b',mew=2,ms=8)
    
    if np.all(abs(z) != np.inf):
        plt.contour(x1,x2,z,levels=10**(np.arange(-20.,1,3)),zoder=100)

def Evaluation(y_val, p_val):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    for epsilon in np.linspace(1.01*min(p_val), max(p_val), 1000):
        
        prediction = (p_val < epsilon)
        tp = np.sum((prediction == y_val) & (y_val == 1))
        fp = np.sum((prediction == 1) & (y_val == 0))
        fn = np.sum((prediction == 0) & (y_val == 1))
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        F1 = 2 * prec * rec /(prec + rec)
        
        if F1 > bestF1:
            bestF1 =F1
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1