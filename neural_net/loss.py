from .utils import *

class binaryCrossEntropy:
    def __str__(self) -> str:
        return 'binaryCrossEntropy'
    def __init__(self,y,p):
        self.y = y
        _,p = p[0]
        self.p = p#.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
        return self.out
    def prime(self,clip=False):
        self. pr = -(self.y/self.p - (1-self.y)/(1-self.p))
        if clip: 
            m,M = non_infinite(self.pr).min(),non_infinite(self.pr).max()
            self.pr = self.pr.clip(m,M)
        return self.pr
class CrossEntropy:
    def __str__(self) -> str:
        return 'CrossEntropy'
    def __init__(self,y,p):
        self.y = y
        _,p = p[0]
        self.p = p#.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = -(self.y*numpy.log(self.p)).sum()/self.p.shape[0] 
        return self.out
    def prime(self,clip=False):
        self. pr = -(self.y/self.p - (1-self.y)/(1-self.p)) #-(self.y/self.p )
        if clip: 
            m,M = non_infinite(self.pr.ravel()).min(),non_infinite(self.pr.pr.ravel()).max()
            self.pr = self.pr.clip(m,M)
        return self.pr
class MSE:
    def __str__(self) -> str:
        return 'MSE'
    def __init__(self,y,p):
        self.y = y
        _,p = p[0]
        self.p = p#.clip(1e-8)#for safety to avoid zero division error
    def eval(self):
        self.out = ((self.y-self.p)**2).mean()
        return self.out
    def prime(self,clip=False):
        self. pr = -2*(self.y-self.p)
        return self.pr