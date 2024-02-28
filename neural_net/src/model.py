
from .utils import unfold
from .db import DBmanager,get_instance,update_instance,tables

class Define(DBmanager):

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def id(self):
        return self._id 
    
    @id.setter
    def id(self,loc):
        loc = unfold(loc)
        self._id = {'id':id(self),f'{str(self)}_id':id(self),'name':repr(self),**loc}
        if not hasattr(self,'table'):
            self.table = get_instance(self)
            self.add_table(self.table)
        else:
            update_instance(self)
            self.commit()

    def __getitem__(self,ix):
        return self.id[ix]
    def __setitem__(self,ix,val):
        self.id[ix] = val
    
    @property
    def get(self):
        return self.id.get


    def __add__(self,loc):
        self.id = loc
        self.c = 0
        class func:
            def __init__(self,_):...
        self.init_method = self.get('Layer_init_method',func)
        self.func = self.get('func',func)
        self.func = self.func(self.id)
        self['steps'] = self.get('steps',[])
        parent ={ f'{str(self)}_id': self['id']}
        for step in self : 
            step.id = {**step.id,**parent}

    def __iter__(self): return self
 
    def __len__(self):
        return len(self['steps'])
    
    def __next__(self):
        if self.c<len(self):
            self.c += 1
            return self['steps'][self.c-1]
        self.c = 0
        raise StopIteration   
    

    def predict(self,X):
        self.out = X
        for step in self:
            self.out = step.func.compute(self.out)
        return self.out

    def update(self,Δ) :
        for step in self['steps'][::-1]:
            Δ = step.func.grad(Δ)
        return Δ

    def compute_store(self):
        value = self.compute(self.y,self.p)
        self.commit()
        del(self.table)
        self + {**self.id,**locals()}
        return value
    
    def updateW(self):
        for obj in Neurons.with_weights:
            for i,r in enumerate(obj.Wtables):
                for j,table in enumerate(r):
                    setattr(table,'value',obj.W[i,j])

class Layer(Define):

    def __str__(self) -> str:
        return 'Layer'  

    
class Neurons(Define):
    with_weights = []

    def instantiateW(self):
        table,cols = tables['Weight']
        self.Wtables = []
        for i,r in enumerate(self.W):
            instances = []
            for j,e in enumerate(r):
                instances += [

                    table(Weight_id=f'{i}_{j}',
                          value=e,
                          Neurons_id=self['id']
                          )
                ]
            self.Wtables += [instances]
            instances = []
        Neurons.with_weights += [self]


    def storeW(self):
        for row in self.Wtables:
            for table in row:
                self.add_table(table)

    def __str__(self) -> str:
        return 'Neurons'
    
    def __sub__(self,Δ):
        self.W -= Δ

    def n(self) : return self.X.shape[0]


    def grad(self,Δ):
        self.Δ = self.pr()*Δ
        return self.Δ  


class Cost(Define):

    def clip(self):
        ε = 1e-7
        self.p =self.p.clip(ε,1-ε)

    def __str__(self) -> str:
        return 'Cost'
    


class Metrics(Define):

    def __str__(self) -> str:
        return 'Metrics'
    

class Architecture(Define):

    def __str__(self) -> str:
        return 'Architecture'