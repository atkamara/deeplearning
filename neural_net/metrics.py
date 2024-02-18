from .utils import *

class accuracy:
    def __init__(self,y,p):
        self.y = y
        self.p = p
    def eval(self):
        self.out = (self.y==self.p).sum()/len(self.y)
        return self.out

        