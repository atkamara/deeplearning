"""
    This modules provides Layer classes

    - `Fullyconnected` 
    - `Activation` 

"""
from .model import Layer
from .activation import Σ


class Fullyconnected(Layer):
    """
    A fully connected neural network layer.

    This layer takes an input vector and transforms it linearly using a weights matrix.
    The product is then subjected to a non-linear activation function.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features .
        init_method (callable): function that initializes weights and takes in as parameters func(n_in,n_out) -> array.shape = (n_in +1, n_out)
        func (callable): default is :func:`~activation.Σ`
        
    """
    def __init__(self,n_in :int,n_out:int,init_method:callable,func:callable=Σ) -> None:
        self + locals()

class Activation(Layer):
    """
    Activation Layer.

    This layer handles activation for a given activation function

    Args:
        func (callable): an activation function like :func:`~activation.σ`

    """
    def __init__(self,func,*kargs) -> None:
        self + locals()