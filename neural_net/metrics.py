from .utils import numpy
from .model import Metrics


class accuracy(Metrics):

    def __init__(self,threshold = .5) -> None:
        self.threshold = threshold

    def compute(self,y,p):

        if y.shape[1]>1:

            p = p.argmax(axis=1)
            y = y.argmax(axis=1)
        else:

            p = (p>self.threshold) + 0

        self.y,self.p = y,p
        return ((self.y==self.p).sum()/len(self.y)).round(4)

class Empty:
    def __repr__(self):return ''
    def __str__(self):return ''
    def __init__(self):...
    def compute(self,y,p): return ''