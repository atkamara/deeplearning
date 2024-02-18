from .utils import *

class binaryCrossEntropy:
    def __init__(self,y,p):
        self.y = y
        self.p = p.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
        return self.out
    def prime(self):
        self. pr = -(self.y/self.p - (1-self.y)/(1-self.p))
        return self.pr
class CrossEntropy:
    def __init__(self,y,p):
        self.y = y
        self.p = p.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = -(self.y*numpy.log(self.p)).sum()/self.p.shape[0] 
        return self.out
    def prime(self):
        self. pr = -(self.y/self.p )/self.p.shape[0]
        return self.pr
class MSE:
    def __init__(self,y,p):
        self.y = y
        self.p = p.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = ((self.y-self.p)**2).mean()
        return self.out
    def prime(self):
        self. pr = -2*(self.y-self.p)
        return self.pr