from .utils import *

def zeros(n_in,n_out,k=1):
    return numpy.zeros((n_in+1,k))

def XHsigmoid(n_in,n_out,k=1):
    r = (6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+1,k))