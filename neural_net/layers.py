from .activation_funcs import *
from .model import Layer,get_class_def


class fullyconnected(Layer):

    def __str__(self):
        return 'fullyconnected'
    
    def __init__(self,n_in,n_out,init_method=zeros):
        self.id   = get_class_def(self,locals())
        self.outfuncs = [Σ(layer=self.id) for _ in range(self.id['n_out'])]
        self.not_stored = True

  
    
class activation(Layer):

    def __str__(self):
        return 'activation'
    
    def __init__(self,n_in,func):
        self.id   = get_class_def(self,locals())
        self.id['n_out'] = n_in
        self.outfuncs = [self.id['func'](layer=self.id) for _ in range(self.id['n_in'])]
        self.not_stored = True
