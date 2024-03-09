"""
    This module provides initialization functions

    - `zeros(n_in: int, n_out: int)` - Initializes a weight matrix with zeros
    - `XHsigmoiduniform` - AA function representing weight initialization using Xavier (Glorot) initialization for sigmoid activation functions.
    - `XHReluuniform` - A function representing weight initialization using Xavier (Glorot) initialization for Rectified linear unit(RELU) activation functions.

"""
from .utils import numpy,Literal

def zeros(n_in: int, n_out: int,biais: bool=True) -> numpy.ndarray:
    """
    Initializes a weight matrix with zeros.

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units.
        biais (bool): if True adds biais weights

    Returns:
        numpy.ndarray: Weight matrix of shape (n_in + 1, n_out).
    """
    return numpy.zeros((n_in+biais,n_out))

class XavierHe:
    """
    This class implements Xavier Glorot and He initializations.(source Hands on ML)

    ![png](static/xahe.png)
    
    Attributes:
        random (dict): contains generators of random values by distribution. 
        activation (str): Name of activation.

    Args:
        distribution (str["Uniform","Normal"]): Name of distribution.
        activation (str["Sigmoid","Tanh","ReLU"]): Name of activation.
    
    Attributes:
        init_func : func(n_in,n_out,biais=True) for generating random values 
    
    """
    random = {
        "Uniform": lambda r,n_in,n_out,biais : numpy.random.uniform(-r,r,size=(n_in+biais,n_out)),
        "Normal": lambda σ,n_in,n_out,biais : numpy.random.normal(scale=σ,size=(n_in+biais,n_out))
    }
    _weight = {
        "Sigmoid": 1,
        "Tanh": 4,
        "ReLU": 2**.5

    }
    default_values = {

        "Uniform" : lambda n_in,n_out : (6/(n_in+n_out))**.5,
        "Normal" : lambda n_in,n_out : (2/(n_in+n_out))**.5,

    }

    @property
    def weight(self):
        return XavierHe._weight.get(self.activation)
    
    @property
    def param(self):
        return XavierHe.default_values.get(self.distribution)
    
    @property
    def init_func(self):
        gen = XavierHe.random.get(self.distribution)
        eq = lambda n_in,n_out,biais=True : gen(self.weight*self.param(n_in,n_out),n_in,n_out,biais)
        return eq
    
    def __init__(self,distribution:Literal["Uniform","Normal"],activation:Literal["Sigmoid","Tanh","ReLU"]) -> None:
        self.activation = activation
        self.distribution = distribution        


def XHsigmoiduniform(n_in: int, n_out: int,biais: bool=True) -> numpy.ndarray:
    """
    A function representing weight initialization using Xavier (Glorot) initialization
    for sigmoid activation functions.

    Attributes:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).
        biais (bool): if True adds biais weights
    
    returns:
        numpy.ndarray : array of random values
    """
    r = (6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+biais,n_out))

def XHReluuniform(n_in: int, n_out: int,biais: bool=True) -> numpy.ndarray:
    """
    A function representing weight initialization using Xavier (Glorot) initialization
    for Rectified linear unit(RELU) activation functions.

    Args:
        n_in (int): Number of input units.
        n_out (int): Number of output units (neurons).
        biais (bool): if True adds biais weights

    returns:
        numpy.ndarray : array of random values
    """
    r = 2**.5*(6/(n_in+n_out))**.5
    return numpy.random.uniform(low=-r,high=r,size=(n_in+biais,n_out))