from .activation_funcs import *


class fullyconnected(GraphManager):
    def __init__(self,n_out,init_method=zeros,In=None,k=1):
        super().__init__()
        self.n_out = n_out
        self.__In = [Σ((id(In),In),init_method,self.n_out,k)]*self.n_out
    @property
    def In(self):
        return self.__In
    @In.setter
    def In(self,v):
        v = dict(v)
        for s in self.__In:
            s.In = (tuple(v.keys()),numpy.concatenate(list(v.values()),axis=1))
    def __len__(self):
        return len(self.In)
    def eval(self):
        self.out = [s.eval() for s in self.In]
        return self.out
    def update(self,Δnext):
        self.Δ =  {v.In[0] : v.update(Δnext[v.outid]) for v in self.In }
        oldks = list(self.Δ)
        for k in oldks:
            if hasattr(k,'__iter__'):
                v = self.Δ[k]
                del(self.Δ[k])
                self.Δ.update(dict(zip(k,v)))
        return self.Δ    
class activation(GraphManager):
    def __init__(self,func=σ,loop=True):
        super().__init__()
        self.func = func
        self.loop = loop
    @property
    def In(self):
        return self.__In
    @In.setter
    def In(self,v):
        if self.loop:
            self.__In = [self.func(s) for s in v]
        else:
            v = dict(v)
            k,v=tuple(v.keys()),numpy.concatenate(list(v.values()),axis=1)
            self.__In = [self.func((k,v))]
    def __len__(self):
        return len(self.In)
    def eval(self):
        self.out = [act.eval() for act in self.In]
        return self.out
    def update(self,Δnext) :
        self.Δ = {act.In[0] : act.update(Δnext[act.outid]) for act in self.In }
        if not self.loop:
             keys = list(self.Δ.keys())[0]
             self.Δ = {k : self.Δ[keys][:,ix].reshape(-1,1) for ix,k in enumerate(keys)}
        return self.Δ        