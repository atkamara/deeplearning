"""
    This modules provides neural network architectures
    Currently available are

    - `Sequential` - Sequential linear net architecture

"""
from .model import Architecture,Layer,Cost,Metrics
from .pipeline import Batch

from .utils import tqdm,numpy
from .metrics import Empty


class Sequential(Architecture):
    

    def __init__(self,steps: list[Layer],cost: Cost) -> None:
        """
        Initialize a Sequential class.

        Args:
            steps (List[Layer]): A list of Layer objects representing the steps.
            cost (Cost): A Cost object for computing cost information.

        Example:
            layer1 = Fullyconnected(2,50,init_funcs.zeros)
            layer2 = Activation(activation.LeakyReLU)
            my_cost = binaryCrossEntropy
            my_instance = Sequential(steps=[layer1, layer2], cost=my_cost)
        """
        self + locals()
        self['cost'] = self['cost'](self['id'])
        self.commit()

    
    def train(self,X:numpy.array=None,y:numpy.array=None,batch:Batch=None,epochs: int = 100, α: float = 0.001,metrics : Metrics=Empty ) -> None:
        """
        Trains a neural network model using sequential architecture

        Args:
            X (numpy.array): Matrix of training features with shape (n, k), where n is the number of samples
                              and k is the number of features.
            y (numpy.array): Target variable with shape (n, 1).
            batch (Optional[Batch]): Optional Batch object that generates batches from the training data.
            epochs (int): Maximum number of training epochs.
            α (float): Learning rate (step size for weight updates).
            metrics (Metrics): Metrics object that computes evaluation metrics (e.g., accuracy).

        Example:
            from neural_net import *
            # generate your training data
            >>> n,k = 5000,2
            >>> X_train = numpy.random.uniform(-100,100,size=(n,k))
            >>> y_train =( (X_train[:, 0]**2 + X_train[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0 
            >>> NN = architecture.Sequential(
                     [

                       layers.Fullyconnected(2,50,init_funcs.XHsigmoiduniform) ,
                       layers.Activation(activation.σ),
                       layers.Fullyconnected(50,1,init_funcs.XHsigmoiduniform) ,
                       layers.Activation(activation.σ),


                    ],
                    cost = cost.binaryCrossEntropy
                )
            >>> NN.train(X_train, y_train,metrics=metrics.accuracy))
        """
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





        