from .db import *
from .utils import *

class Sequential(GraphManager):

    def __init__(self,steps,db=None):
        super().__init__()
        DBmanager().start(db)
        self.steps = steps

    def eval(self):
        self.out  = self.steps[0].eval()
        for step in self.steps[1:]:
            step.In = self.out
            step.eval()
            self.out = step.out
            self.outid = self.out[0][0]
        return self.out
    
    def predict(self,X):
        self.steps[0].In = [(id(X),X)]
        self.out = self.eval()
        return self.out
    
    def train(self,X,y,α,n_epochs,metrics,batch_size=None):

        batchix = get_batchix(len(y),batch_size)

        for e in range(n_epochs):
            for ix in batchix:
                new_y,new_X = y[ix,:],X[ix,:]
                self.predict(new_X)
                self.update({self.outid:new_y})
            print('epoch',e,str(metrics),metrics(new_y,self.out).eval())

    def update(self,Δnext_or_ynext):
        self.Δnext_or_ynext = Δnext_or_ynext
        for step in self.steps[::-1]:
            self.Δnext_or_ynext = step.update(self.Δnext_or_ynext)
        return self