from .db import *

class Layer(DBmanager,GraphManager):
    @property
    def In(self):
        return self._In
    @In.setter
    def In(self,In):
        
        if self.not_stored :
            self.insert_db(*SQL.layers(self))
            self.not_stored = False
        self._In = dict(In)
        for i,s in enumerate(self.outfuncs):
            if str(self)=='fullyconnected':
                s.In = (tuple(self.In.keys()),numpy.concatenate(list(self.In.values()),axis=1))
            elif str(self)=='activation':
                s.In = list(self.In.items())[i]

    def __len__(self):
        return len(self.In)
    def eval(self):
        self.out = [s.eval() for s in self.outfuncs]
        return self.out
    def update(self,Δnext):
        self.Δ =  {s.In[0] : s.update(Δnext[s.outid]) for s in self.outfuncs }
        oldks = list(self.Δ)
        for k in oldks:
            if hasattr(k,'__iter__'):
                v = self.Δ[k]
                del(self.Δ[k])
                self.Δ.update({i:v for i in k})
        return self.Δ  
    
class neuron(DBmanager,GraphManager):
    @property
    def In(self):
        return self._In
    @In.setter
    def In(self,val):
        self._In = val
        if self.not_stored:
            self.id['n_in'] = self.In[1].shape[1]
            self.insert_db(*SQL.neuron(self))
            self.not_stored = False
        if str(self)=='Linear':
            if self.not_stored_init_w : 
                self.insert_db(*SQL.weights(self))
                self.not_stored_init_w=False
    def eval(self):
        self.outid,self.out = self.add_to_graph((self.In[0],self.getout())) 
        return (self.outid,self.out)