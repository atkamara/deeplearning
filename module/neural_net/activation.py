"""
    This module provides classes for several types of activation functions

    - `Σ` - Linear combination of weights and biases
    - `σ` - sigmoid activation
    - `Softmax`- Softmax activation
    - `LeakyReLU`- Leaky rectified linear unit activation

"""
from .utils import numpy
from .model import Neurons,Layer

class Σ(Neurons):
    r"""
    A class representing a linear combination operation.

    $$
    \mathrm{\mathit{H}}(z) = z.w + b
    $$

    where w is weights vector and b is bias

    Attributes:
        W (numpy.array): Weight matrix of shape (k+1, n_out). +1 for bias

    Methods:
        compute(X):
            Computes the linear combination of input matrix X and bias vector using weight matrix W.

        pr():
            Computes the derivative of the linear equation with respect to W (matrix X itself).

        grad(Δ):
            Updates weights self.W and computes the new gradient Δ for backpropagation.
        Xb():
            Concatenates X matrix with a vector of ones

    """

    def __init__(self,Layer:Layer=None) -> None:
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
class Tanh(Neurons):
    r"""
    A class representing the Hyperbolic Tangent activation function.

    $$
    \tanh(z)={\frac{{\rm{e}}^{z}-{\rm{e}}^{-z}}{{\rm {e}}^{z}+{\rm {e}}^{-z}}}={\frac{{\rm {e}}^{2z}-1}{{\rm{e}}^{2z}+1}}={\frac{1-{\rm{e}}^{-2z}}{1+{\rm {e}}^{-2z}}}
    $$

    $$
    Tanh=2\sigma(2z) - 1
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
            Computes the Tanh activation for input matrix X.

        pr():
            Computes the derivative of the Tanh function.

    """
    def __init__(self,Layer:Layer=None) -> None:
        self + locals()
    
    def pr(self) -> numpy.array:
        r"""
        Computes the derivative of the Tanh function.

        $$
         \tanh '={\frac {1}{\cosh ^{2}}}=1-\tanh ^{2}
        $$

        Returns:
            numpy.array: Derivative matrix.
        """
        return 1-self.preds**2
    
    def compute(self,X:numpy.array) -> numpy.array:
        r"""
        Computes the Tanh activation for input matrix X.

        $$
        Tanh(X)=2\sigma(2X) - 1
        $$

        where $\sigma$ is defined as follows:

        $$
         \sigma (X)={\frac {1}{1+e^{-X}}}
        $$

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Tanh activation result of shape (n, n_out).
        """
        self.X = X
        σ = lambda z : 1/(1+numpy.exp(-z))
        self.preds = 2*(2*σ(2*self.X)-1)
        return self.preds
class σ(Neurons):
    r"""
    A class representing the sigmoid activation function.

    $$
    \sigma(z)={\frac{1}{1+e^{-z}}}={\frac{e^{z}}{1+e^{z}}}=1-\sigma(-z)
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
                Computes the sigmoid activation for input matrix X.

        pr():
            Computes the derivative of the sigmoid function.
    """
    def __init__(self,Layer:Layer=None) -> None:
        self + locals()
    
    def pr(self) -> numpy.array:
        r"""
        Computes the derivative of the sigmoid function.

        $$
        {\begin{aligned}\sigma'(z)&={\frac {e^{z}\cdot (1+e^{z})-e^{z}\cdot e^{z}}{(1+e^{z})^{2}}}\\&={\frac {e^{z}}{(1+e^{z})^{2}}}\\&=\left({\frac {e^{z}}{1+e^{z}}}\right)\left({\frac {1}{1+e^{z}}}\right)\\&=\left({\frac {e^{z}}{1+e^{z}}}\right)\left(1-{\frac {e^{z}}{1+e^{z}}}\right)\\&=\sigma(z)\left(1-\sigma(z)\right)\end{aligned}}
        $$

        Returns:
            numpy.array: Derivative matrix.
        """
        return self.preds*(1-self.preds)
    
    def compute(self,X:numpy.array) -> numpy.array:
        r"""
        Computes the sigmoid activation for input matrix X  using vectorization with numpy.

        $$
         \sigma (X)={\dfrac {1}{1+e^{-X}}}
        $$

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Sigmoid activation result of shape (n, n_out).
        """
        self.X = X
        self.preds = 1/(1+numpy.exp(-self.X))
        return self.preds
 
class Softmax(Neurons):
    r"""
    A class representing the softmax activation function.
    
    $$
    \sigma(\mathbf {z_{i}} )=\frac {e^{z_{i}}}{\sum _{j=1}^{k}e^{z_{j}}}\\ {\text{ for }}j=1,\dotsc ,k{\text{ features }}{\text{ and }} z_{i}=(z_{i,1},...,z_{i,n}) \text{ for n observations}
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
            Computes the Softmax activation for input matrix X.

        pr():
            Computes the derivative of the Softmax function.

    """   
    def __init__(self,Layer:Layer=None) -> None:
        self + locals()
    def pr(self) -> numpy.array:
        """
        Computes the derivative of the Softmax function.

        Returns:
            numpy.array: Derivative matrix.
        """
        return self.preds*(1-self.preds)

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the softmax activation for input matrix X using vectorization with numpy.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: Softmax activation result of shape (n, n_out).
        """
        self.X = X
        self.preds = (ex:=numpy.exp(self.X))/ex.sum(axis=1).reshape(-1,1)
        return self.preds
class ELU(Neurons):
    r"""
    A class representing the Exponential Linear Unit (ELU) activation function.

    $$
    \mathrm{\mathit{H}}(z) = \begin{cases}
        z & \text{if } z \geq 0  \\ % & is your "\tab"
        \alpha (e^{z} - 1) & \text{if } z < 0
    \end{cases}
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
            Computes the ELU activation for input matrix X.

        pr():
            Computes the derivative of the ELU function.

    Args:
        α (float): The slope coefficient for negative values (default is 0.001).
   

    """
    def __init__(self,Layer:Layer=None,α=0.001) -> None:
        self + locals()
    
    def pr(self) -> numpy.array: 
        r"""
        Computes the derivative of the ELU function.

        $$
        \mathrm{\mathit{H}}'(z) = \begin{cases}
            1 & \text{if } z \geq 0  \\ % & 
            \mathrm{\mathit{H}}(z) + \alpha & \text{if } z < 0
        \end{cases}
        $$

        Returns:
            numpy.array: Derivative matrix.
        """
        return (neg := self.X < 0)*self.preds + neg*self['α'] + ~neg

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the ELU activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: ELU activation result of shape (n, n_out).
        """
        self.X = X
        self.preds = (neg := self.X<0)*self['α']*(numpy.exp(self.X)-1) + ~neg*self.X
        return self.preds
class ReLU(Neurons):
    r"""
    A class representing the Rectified Linear Unit (ReLU) activation function.

    $$
    \mathrm{\mathit{H}}(z) = \begin{cases}
        z & \text{if } z \geq 0  \\ % & is your "\tab"
        0 & \text{if } z < 0
    \end{cases}
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
            Computes the ReLU activation for input matrix X.

        pr():
            Computes the derivative of the ReLU function.

    Args:
        α (float): The slope coefficient for negative values (default is 0.001).

    """
    def __init__(self,Layer:Layer=None) -> None:
        self + locals()
    
    def pr(self) -> numpy.array: 
        r"""
        Computes the derivative of the ReLU function.

        $$
        \mathrm{\mathit{H}}(z) = \begin{cases}
            1 & \text{if } z \geq 0  \\ % & is your "\tab"
            0 & \text{if } z < 0
        \end{cases}
        $$

        Returns:
            numpy.array: Derivative matrix.
        """
        return (self.X >= 0) + 0 #for casting bool to int

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the ReLU activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: ReLU activation result of shape (n, n_out).
        """
        self.X = X
        self.preds = numpy.maximum(0,self.X)
        return self.preds
class LeakyReLU(Neurons):
    r"""
    A class representing the Leaky Rectified Linear Unit (LeakyReLU) activation function.

    $$
    \mathrm{\mathit{H}}(z) = \begin{cases}
        z & \text{if }  z \geq 0  \\ % & is your "\tab"
        \alpha z & \text{if } z < 0
    \end{cases}
    $$

    Attributes:
        preds: predicted values.

    Methods:
        compute(X):
            Computes the LeakyReLU activation for input matrix X.

        pr():
            Computes the derivative of the LeakyReLU function.

    Args:
        α (float): The slope coefficient for negative values (default is 0.001).
    """
    def __init__(self,Layer:Layer=None,α:float=.001) -> None:
        self + locals()
    
    def pr(self) -> numpy.array: 
        r"""
        Computes the derivative of the LeakyReLU function.
        $$
        \mathrm{\mathit{H}}'(z) = \begin{cases}
        1 & \text{if } z \geq 0  \\ % &
        \alpha & \text{if } z < 0
        \end{cases}
        $$

        Returns:
            numpy.array: Derivative matrix.
        """
        return (neg:=self.X < 0)*self['α'] + ~neg

    def compute(self,X:numpy.array) -> numpy.array:
        """
        Computes the LeakyReLU activation for input matrix X.

        Args:
            X (numpy.array): Input matrix of shape (n, k).

        Returns:
            numpy.array: LeakyReLU activation result of shape (n, n_out).
        """
        self.X = X
        self.preds = (neg := self.X<0)*self['α']*self.X + ~neg*self.X
        return self.preds