from .utils import numpy

def zeros(n_in,n_out):
    return numpy.zeros((n_in+1,n_out))

def XHsigmoiduniform(n_in,n_out):
    r = (6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+1,n_out))

def XHReluuniform(n_in,n_out):
    r = 2**.5*(6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+1,n_out))