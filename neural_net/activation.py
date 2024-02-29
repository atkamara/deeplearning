"""
    This modules provides classes for several types of activation functions

    - `Σ` - Linear combination of weights and biases
    - `σ` - sigmoid activation
    - `Softmax`- Softmax activation
    - `LeakyReLU`- Leaky rectified linear unit activation

"""
from .utils import numpy
from .model import Neurons,Layer

class Σ(Neurons):
    """
    A class representing a linear combination operation.

    Attributes:
        W (numpy.array): Weight matrix of shape (k+1, n_out).

    Methods:
        compute(X):
            Computes the linear combination of input matrix X and bias vector using weight matrix W.

        pr():
            Computes the derivative of the linear equation with respect to W (matrix X itself).

        grad(Δ):
            Updates weights self.W and computes the new gradient Δ for backpropagation.

    """

    def __init__(self,Layer:Layer) -> None:
        self + locals()
        self.W = self.init_method(self['Layer_n_in'],self['Layer_n_out'])
        self.Xb = lambda : numpy.c_[self.X,numpy.ones((self.n(),1))]
        self.instantiateW()
        self.storeW()
    def pr(self)-> numpy.array:
        """
        Computes the derivative of the linear equation (matrix itself).

        Returns:
            numpy.array: Derivative matrix.
        """
        return self.Xb

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the linear combination of input matrix X and bias vector using weight matrix self.W.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Linear combination result of shape (n, n_out).
        """
        self.X = X
        return self.Xb().dot(self.W) 
    
    def grad(self,Δ :numpy.array) -> numpy.array:
        """
        Updates weights self.W and computes the gradient for backpropagation.

        Args:
            Δ (numpy.array): Gradient from next activation.
        """
        self   - (self.pr().T.dot(Δ))/self.n()
        self.Δ = Δ.dot(self.W[:-1,:].T) #-1 to remove biais
        return self.Δ        

class σ(Neurons):
    """
    A class representing the sigmoid activation function.

        Attributes:
            None

        Methods:
            compute(X):
                Computes the sigmoid activation for input matrix X.

            pr():
                Computes the derivative of the sigmoid function.

            grad(Δ):
                Computes the gradient for backpropagation.

        Args:
            None
        """
    def __init__(self,Layer:Layer) -> None:
        self + locals()
    
    def pr(self) -> numpy.array:
        """
        Computes the derivative of the sigmoid function.

        Returns:
            numpy.array: Derivative matrix.
        """
        return self.probs*(1-self.probs)
    
    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the sigmoid activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Sigmoid activation result of shape (n, n_out).
        """
        self.X = X
        self.probs = 1/(1+numpy.exp(-self.X))
        return self.probs
 
class Softmax(Neurons):
    """
        A class representing the softmax activation function.

        Attributes:
            None

        Methods:
            compute(X):
                Computes the softmax activation for input matrix X.

            pr():
                Computes the derivative of the softmax function.

            grad(Δ):
                Computes the gradient for backpropagation.

        Args:
            None
        """   
    def __init__(self,Layer:Layer) -> None:
        self + locals()

    def pr(self) -> numpy.array:
        """
        Computes the derivative of the sigmoid function.

        Returns:
            numpy.array: Derivative matrix.
        """
        return self.probs*(1-self.probs)

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the softmax activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Softmax activation result of shape (n, n_out).
        """
        self.X = X
        self.probs = (ex:=numpy.exp(self.X))/ex.sum(axis=1).reshape(-1,1)
        return self.probs

class LeakyReLU(Neurons):
    """
    A class representing the Leaky Rectified Linear Unit (LeakyReLU) activation function.

    Attributes:
        leak (float): The slope coefficient for negative values.

    Methods:
        compute(X):
            Computes the LeakyReLU activation for input matrix X.

        pr():
            Computes the derivative of the LeakyReLU function.

        grad(Δ):
                Computes the gradient for backpropagation.

    Args:
        alpha (float): The slope coefficient for negative values (default is 0.001).
    """
    def __init__(self,Layer:Layer,leak:float=.001) -> None:
        self + locals()
    
    def pr(self) -> numpy.array: 
        """
        Computes the derivative of the LeakyReLU function.

        Returns:
            numpy.array: Derivative matrix.
        """
        return(neg:=self.X < 0)*self['leak'] + ~neg

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the LeakyReLU activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: LeakyReLU activation result of shape (n, n_out).
        """
        self.X = X
        return numpy.maximum(self['leak']*self.X,self.X)