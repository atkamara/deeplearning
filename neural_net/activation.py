from .utils import numpy
from .model import Neurons

class Σ(Neurons):
    
    def __init__(self,Layer) -> None:
        self + locals()
        self.W = self.init_method(self['Layer_n_in'],self['Layer_n_out'])
        self.Xb = lambda : numpy.c_[self.X,numpy.ones((self.n(),1))]
        self.pr = self.Xb
        self.instantiateW()
        self.storeW()

    def compute(self,X):
        self.X = X
        return self.Xb().dot(self.W) 
    
    def grad(self,Δ):
        self   - (self.pr().T.dot(Δ))/self.n()
        self.Δ = Δ.dot(self.W[:-1,:].T) #-1 to remove biais
        return self.Δ        

class σ(Neurons):
    
    def __init__(self,Layer) -> None:
        self + locals()
        self.pr = lambda : self.probs*(1-self.probs)
    
    def compute(self,X):
        self.X = X
        self.probs = 1/(1+numpy.exp(-self.X))
        return self.probs


 
    
class Softmax(Neurons):
    
    def __init__(self,Layer) -> None:
        self + locals()
        self.pr = lambda : self.probs*(1-self.probs)

    def compute(self,X):
        self.X = X
        self.probs = (ex:=numpy.exp(self.X))/ex.sum(axis=1).reshape(-1,1)
        return self.probs



class LeakyReLU(Neurons):
    
    def __init__(self,Layer,leak=.001) -> None:
        self + locals()
        self.pr = lambda: (neg:=self.X < 0)*self['leak'] + ~neg

    def compute(self,X):
        self.X = X
        return numpy.maximum(self['leak']*self.X,self.X)

