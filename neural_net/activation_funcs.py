from .init_funcs import *

class Σ(GraphManager):
    def __init__(self,In,init_method=zeros,n_out=None,k=1):
        super().__init__()
        self.In = In
        self.init_method=init_method
        self.layer_nout = n_out
        self.k = k
    def eval(self):
        if not hasattr(self,'w'):
            self.w = self.init_method(self.In[1].shape[1],self.layer_nout,self.k)
        self.b = numpy.ones((self.In[1].shape[0],1))
        out = lambda : numpy.c_[self.In[1],self.b].dot(self.w)
        self.outid,self.out = self.add_to_graph((self.In[0],out())) 
        return (self.outid,self.out)
    def prime(self):
        self.pr = numpy.c_[self.In[1],self.b]
        return self.pr
    def update(self,Δnext):
        self.Δ = (self.prime().T.dot(Δnext)).reshape(-1,self.k)
        self.Δ /= self.In[1].shape[0]
        self.w -= self.Δ
        return self.Δ.sum(axis=1).reshape(-1,1)

class σ(GraphManager):
    def __init__(self,In=None):
        super().__init__()
        self.In = In
    def eval(self):
        out = lambda: 1/(1+numpy.exp(-self.In[1]))
        self.outid,self.out = self.add_to_graph((self.In[0],out())) 
        return (self.outid,self.out)
    def prime(self):
        self.p = self.out*(1-self.out)
        return self.p
    def update(self,Δnext):
        self.Δ = Δnext*self.prime()
        return self.Δ   
class LeakyRelu(GraphManager):
    def __init__(self,In=None,α=0.01):
        super().__init__()
        self.In = In
        self.α = α
    def eval(self):
        out = lambda : numpy.maximum(self.α*self.In[1],self.In[1])
        self.outid,self.out = self.add_to_graph((self.In[0],out())) 
        return (self.outid,self.out)
    def prime(self):
        I1 = numpy.ones(self.out.shape)
        self.p = numpy.maximum(self.α*I1,I1)
        return self.p
    def update(self,Δnext):
        self.Δ = Δnext*self.prime()
        return self.Δ    
class heaviside(GraphManager):
    def __init__(self,In=None):
        super().__init__()
        self.In = In
    def eval(self):
        out = lambda: numpy.where(self.In[1]>=0,1,0)
        self.outid,self.out = self.add_to_graph((self.In[0],out())) 
        return (self.outid,self.out)
    def prime(self):
        ...
    def update(self,Δnext):
        self.Δ = Δnext*self.prime()
        return self.Δ   
class Softmax(GraphManager):#normalized exponential
    def __init__(self,In=None):
        super().__init__()
        self.In = In
    def eval(self):
        out = lambda: numpy.exp(self.In[1])/numpy.exp(self.In[1]).sum(axis=1).reshape(-1,1)
        self.outid,self.out = self.add_to_graph((self.In[0],out())) 
        return (self.outid,self.out)
    def prime(self):
        self.p = self.out*(1-self.out)
        return self.p
    def update(self,Δnext):
        self.Δ = Δnext*self.prime()
        return self.Δ    