from .utils import *

class accuracy:
    def __init__(self,y,p,threshold = .5):
        _,p = p[0]
        self.threshold = threshold
        p = (p>self.threshold) + 0
        self.y,self.p = y,p
    def __str__(self):
        return 'accuracy'
    def eval(self):
        if self.y.shape[1]>1:
            self.p = self.p.argmax(axis=1)
            self.y = self.y.argmax(axis=1)
        self.out = (self.y==self.p).sum()/len(self.y)
        return self.out

        