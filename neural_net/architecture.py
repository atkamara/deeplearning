from .db import *
from .utils import *

class Sequential(GraphManager):

    def __init__(self,steps,cost_func,db=None):
        super().__init__()
        self.db_path = DBmanager().start(db)
        self.steps = steps
        self.cost_func = cost_func

    def eval(self,X):
        self.steps[0].In = [(id(X),X)]
        self.out  = self.steps[0].eval()
        for step in self.steps[1:]:
            step.In = self.out
            step.eval()
            self.out = step.out
            self.outid = self.out[0][0]
        return self.out
    
    def predict(self,X):
        self.eval(X)
        return self.out
    
    def train(self,X,y,n_epochs=100,α=1,metrics=None,batch_size=None):

        batchix = get_batchix(len(y),batch_size)

        for e in range(n_epochs):

            for ix in batchix:
                new_y,new_X = y[ix,:],X[ix,:]
                new_pred = self.predict(new_X)
                c = self.cost_func(new_y,new_pred)
                Δ = α*c.prime()
                self.update({self.outid:Δ})
            if metrics:
                m = metrics(new_y,new_pred)
                print('epoch',e,str(c),c.eval(),str(m),m.eval())


    def update(self,Δ):
        self.Δ = Δ
        for step in self.steps[::-1]:
            self.Δ = step.update(self.Δ)
        return self