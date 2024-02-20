from .init_funcs import *
from .loss import binaryCrossEntropy,CrossEntropy 
from .model import neuron,get_class_def,SQL


class Σ(neuron):

    def __str__(self):
        return 'Linear'
    
    def __init__(self,layer=None):
        self.id = get_class_def(self,locals())
        self.w = self.id['layer']['init_method'](self.id['layer']['n_in'])
        self.not_stored = True
        self.not_stored_init_w = True

    def getout(self):
        self.b = numpy.ones((self.In[1].shape[0],1))
        return numpy.c_[self.In[1],self.b].dot(self.w)
    
    def prime(self):
        self.pr = numpy.c_[self.In[1],self.b]
        return self.pr
    
    def update(self,Δnext):
        self.Δ = (self.prime().T.dot(Δnext)).reshape(-1,1)/self.In[1].shape[0]
        self.w -= self.Δ
        self.insert_db(*SQL.weights(self))
        return self.Δ.sum(axis=1).reshape(-1,1)

class σ(neuron):
    def __str__(self):
        return 'sigmoid'
    
    def __init__(self,layer=None,cost_func=binaryCrossEntropy):
        self.id = get_class_def(self,locals())
        self.not_stored = True

    def getout(self): return 1/(1+numpy.exp(-self.In[1]))

    def update(self,y):
        self.Δ = self.id['cost_func'](y,self.out).prime()
        return self.Δ   



class Softmax(neuron):#normalized exponential
    def __str__(self):
        return 'Softmax'
    
    def __init__(self,In=None,layer=None,cost_func=binaryCrossEntropy):
        self.id = get_class_def(self,locals())
        self.__In = In
        self.not_stored = True

    def getout(self): return numpy.exp(self.In[1])/numpy.exp(self.In[1]).sum(axis=1).reshape(-1,1)

    def update(self,y):
        self.Δ = self.id['cost_func'](y,self.out).prime()
        return self.Δ  