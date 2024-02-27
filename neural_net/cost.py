from .utils import numpy
from .model import Cost


class binaryCrossEntropy(Cost):
    
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
        self. pr = lambda : -(self.y/self.p - (1-self.y)/(1-self.p))
    
    def compute(self,y,p):
        self.y,self.p= y,p.clip(1e-7,1-1e-7)
        return -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()

    
class CrossEntropy(Cost):
    
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
        self.pr = lambda : -(self.y/(p:=self.p.clip(1e-7,1-1e-7))- (1-self.y)/(1-p) ) 

    def compute(self,y,p):
        self.y,self.p = y,p.clip(1e-7,1-1e-7)
        return -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
        

    

class MSE(Cost):
    
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
        self. pr = lambda : -2*(self.y-self.p)

    def compute(self,y,p):
        self.y,self.p = y,p
        return ((self.y-self.p)**2).mean()
