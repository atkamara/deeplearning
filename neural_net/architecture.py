from .model import Architecture
from .utils import tqdm
from .metrics import Empty

class Sequential(Architecture):
    

    def __init__(self,steps,cost) -> None:
        self + locals()
        self['cost'] = self['cost'](self['id'])
        self.commit()

    
    def train(self,X=None,y=None,batch=None,epochs=100,α=1,metrics=Empty):



        Xys = batch or [(X,y)]
        epochs = tqdm.tqdm(range(epochs))
        m = metrics()

        for _ in epochs:

            for X,y in Xys:

                self.out = self.predict(X)
                self['cost'].compute(y,self.out)
                self.update(α*self['cost'].pr())
                
            epochs.set_description(' '.join(map(repr,[
                                            self['cost'],
                                            self['cost'].compute_store().round(4),
                                            m,
                                            m.compute(y,self.out)]))) 
        self.updateW()
        self.commit()      





        