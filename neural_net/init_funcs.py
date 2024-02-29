"""
    This modules provides initialization functions

    - `zeros(n_in: int, n_out: int)` - Initializes a weight matrix with zeros
    - `XHsigmoiduniform` - AA function representing weight initialization using Xavier (Glorot) initialization for sigmoid activation functions.
    - `XHReluuniform` - A function representing weight initialization using Xavier (Glorot) initialization for Rectified linear unit(RELU) activation functions.

"""
from .utils import numpy

def zeros(n_in: int, n_out: int) -> numpy.array:
    """
    Initializes a weight matrix with zeros.

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units.

    Returns:
        numpy.array: Weight matrix of shape (n_in + 1, n_out).
    """
    return numpy.zeros((n_in+1,n_out))

def XHsigmoiduniform(n_in: int, n_out: int) -> numpy.array:
    """
    A function representing weight initialization using Xavier (Glorot) initialization
    for sigmoid activation functions.

    Attributes:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).
    """
    r = (6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+1,n_out))

def XHReluuniform(n_in: int, n_out: int) -> numpy.array:
    """
    A function representing weight initialization using Xavier (Glorot) initialization
    for Rectified linear unit(RELU) activation functions.

    Attributes:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).
    """
    r = 2**.5*(6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+1,n_out))