from .model import Layer
from .activation import Σ


class Fullyconnected(Layer):
    
    def __init__(self,n_in,n_out,init_method,func=Σ) -> None:
        self + locals()

class Activation(Layer):
    
    def __init__(self,func,init_method=None) -> None:
        self + locals()