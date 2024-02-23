from .activation_funcs import *
from .model import Layer,get_class_def


class fullyconnected(Layer):

    def __str__(self):
        return 'fullyconnected'
    
    def __init__(self,n_in,n_out,init_method=zeros,store=True):
        self.id   = get_class_def(self,locals())
        self.outfuncs = [Σ(layer=self.id) for _ in range(self.id['n_out'])]

  
    
class activation(Layer):

    def __str__(self):
        return 'activation'
    
    def __init__(self,n_in,n_out,func,store=True):
        self.id   = get_class_def(self,locals())
        self.outfuncs = [self.id['func'](layer=self.id) for _ in range(self.id['n_out'])]
