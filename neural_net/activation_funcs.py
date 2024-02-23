from .init_funcs import *
from .loss import binaryCrossEntropy,CrossEntropy 
from .model import neuron,get_class_def,SQL


class Σ(neuron):

    def __str__(self):
        return 'Linear'
    
    def __init__(self,layer=None):
        self.id = get_class_def(self,locals())
        self.w = self.id['layer']['init_method'](self.id['layer']['n_in'],self.id['layer']['n_out'])
        self.not_stored = self.id['layer']['store']
        self.not_stored_init_w = self.id['layer']['store']

    def getout(self):
        self.b = numpy.ones((self.In[1].shape[0],1))
        return numpy.c_[self.In[1],self.b].dot(self.w)
    
    def prime(self):
        self.pr = numpy.c_[self.In[1],self.b]
        return self.pr
    
    def update(self,Δ):
        self.Δ = Δ.reshape(-1,1)
        #####breakpoint#######
        self.w -= (self.prime().T.dot(self.Δ))/self.In[1].shape[0]
        self.insert_db(*SQL.weights(self))
        #####resumeflow#######
        self.Δ = self.Δ.dot(self.w.T)
        return self.Δ

class σ(neuron):
    def __str__(self):
        return 'sigmoid'
    
    def __init__(self,layer=None):
        self.id = get_class_def(self,locals())
        self.not_stored = self.id['layer']['store']

    def getout(self): return 1/(1+numpy.exp(-self.In[1]))

    def prime(self):
        self.pr = self.out*(1-self.out)
        return self.pr       

    def update(self,Δ):
        self.Δ = self.prime()*Δ
        return self.Δ   

class LeakyReLU(neuron):
    def __str__(self):
        return 'LeakyReLU'
    
    def __init__(self,leak=.01,layer=None):
        self.id = get_class_def(self,locals())
        self.not_stored = self.id['layer']['store']

    def getout(self): return numpy.maximum(self.id['leak']*self.In[1],self.In[1])

    def prime(self):
        neg = self.In[1] < 0
        self.pr = neg*self.id['leak'] + ~neg
        return self.pr       

    def update(self,Δ):
        self.Δ = self.prime()*Δ
        return self.Δ   




class Softmax(neuron):#normalized exponential
    def __str__(self):
        return 'Softmax'
    
    def __init__(self,In=None,layer=None):
        self.id = get_class_def(self,locals())
        self.not_stored = self.id['layer']['store']

    def getout(self): return numpy.exp(self.In[1])/numpy.exp(self.In[1]).sum(axis=1).reshape(-1,1)

    def prime(self):
        self.pr = self.out*(1-self.out)
        return self.pr       

    def update(self,Δ):
        self.Δ = self.prime()*Δ
        return self.Δ   