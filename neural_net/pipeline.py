from .utils import numpy,pandas

def get_ix(size,obs):  
        batchix = list(range(0,obs,size))
        if batchix[-1]<obs : batchix.append(obs)
        batchix = [slice(low,high) for low,high in zip(batchix,batchix[1:])]
        return batchix

def shuffle(X,y):

    X = pandas.DataFrame(X).sample(frac=1)
    
    y = pandas.DataFrame(y).loc[X.index]

    return X.values,y.values

class Batch:

    def __init__(self,size,obs,X,y) -> None:
        self.getters = lambda ix: (X()[ix,:],y()[ix,:])
        self.i = self.getters(slice(0,10))
        self.ix = get_ix(size,obs) 
        self.c=0

    def __iter__(self): return self
    def __next__(self):
        if self.c<len(self.ix):
            self.c += 1
            return self.getters(self.ix[self.c-1])
        self.c = 0
        raise StopIteration
    
onehot = lambda y : (y==numpy.unique(y))+0




scaler = lambda X : (X-X.mean())/X.std()