"""
    This module provides classes for several types of cost functions

    - `binaryCrossEntropy` 
    - `CrossEntropy` 
    - `MSE`

"""
from .utils import numpy
from .model import Cost


class BinaryCrossEntropy(Cost):
    r"""
    Binary Cross-Entropy Loss.
    $$
    \mathrm{\mathit{Binary\ Cross\ Entropy}}(p, y) = \begin{cases}
    -\log(p) & \text{if } y = 1, \\
    -\log(1-p) & \text{otherwise.}
    \end{cases}
    $$    
    This class computes the binary cross-entropy loss between true labels (y) and predicted probabilities (p).

    Methods:
        - compute(y: numpy.ndarray, p: numpy.ndarray) -> float:
            Computes the binary cross-entropy loss.

        - pr(y: numpy.ndarray, p: numpy.ndarray) -> numpy.ndarray:
            Computes the derivative function values.

    Example:
        ```python
                >>> y_true = numpy.array([[0], [1], [1], [0]])
                >>> predicted_probs = numpy.array([[0.2], [0.8], [0.6], [0.3]])
                >>> bce_loss = binaryCrossEntropy()
                >>> loss_value = bce_loss.compute(y_true, predicted_probs)
                >>> print(f"Binary Cross-Entropy Loss: {loss_value:.4f}")
                Binary Cross-Entropy Loss: 0.3284
                >>> derivative_values = bce_loss.pr()
                >>> print(f"Derivative Function Values: {derivative_values}")
                Derivative Function Values: [ 1.25       -1.25       -1.66666667  1.42857143]
        ```
    """    
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.ndarray:
        """
        Computes the derivative function values  with respet to p.

        Returns:
            numpy.ndarray: Derivative function values.
        """
        return -(self.y/self.p - (1-self.y)/(1-self.p))
    def compute(self,y: numpy.ndarray,p: numpy.ndarray,clip : bool=True) -> float:
        """
        Computes the binary cross-entropy loss.

        Args:
            y (numpy.ndarray): True labels (0 or 1).
            p (numpy.ndarray): Predicted probabilities (between 0 and 1).
            clip (bool): Whether or not to clip predicted values see method clip

        Returns:
            float: Binary cross-entropy loss value.
        """
        self.y,self.p = y,p
        if clip:self.clip()
        return -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
   
class CrossEntropy(Cost):
    r"""
    Cross-Entropy Loss.

    This class computes the cross-entropy loss between true labels (y) and predicted probabilities (p).
    $$
    Cross\ Entropy(p,y) = -\sum _{i}\sum _{j}y_{ij}\log p_{ij}\ 
    $$
    
    Methods:
        - compute(y: numpy.ndarray, p: numpy.ndarray) -> float:
            Computes the cross-entropy loss.

        - pr() -> numpy.ndarray:
            Computes the derivative function values.

    Example:
        ```python
                >>> y_true = numpy.array([[1, 0, 0],
                ...        [0, 1, 0],
                ...        [0, 0, 1],
                ...        [0, 1, 0],
                ...        [1, 0, 0]])
                >>> predicted_probs = numpy.array([[0, 0.6, 0.3],
                ...                                [0.4, 0.2, 0.4],
                ...                                [0.2, 0.3, 0.5],
                ...                                [0.5, 0.1, 0.4],
                ...                                [0.3, 0.4, 0.3]])
                >>> ce_loss = CrossEntropy()
                >>> loss_value = ce_loss.compute(y_true, predicted_probs)
                >>> print(f"Cross-Entropy Loss: {loss_value:.4f}")
                Cross-Entropy Loss: 1.7915
                >>> derivative_values = ce_loss.pr()
                >>> print(f"Derivative Function Values: {derivative_values}")
                Derivative Function Values: array([[-1.00000000e+07,  2.50000000e+00,  1.42857143e+00],
               [ 1.66666667e+00, -5.00000000e+00,  1.66666667e+00],
               [ 1.25000000e+00,  1.42857143e+00, -2.00000000e+00],
               [ 2.00000000e+00, -1.00000000e+01,  1.66666667e+00],
               [-3.33333333e+00,  1.66666667e+00,  1.42857143e+00]])
        ```
    """
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.ndarray:
        """
        Computes the derivative function values  with respet to p .

        Returns:
            numpy.ndarray: Derivative function values.
        """
        left  = (self.y/self.p)
        #right = left.sum(axis=1,keepdims=True)*(f:=((1-self.y)/(1-self.p)))/f.sum(axis=1,keepdims=True)
        right = (1-self.y)/(1-self.p)
        return -(left - right)
    def compute(self,y: numpy.ndarray,p: numpy.ndarray,clip : bool=True) -> float:
        """
        Computes the Cross-entropy loss.

        Args:
            y (numpy.ndarray): True labels (0 or 1).
            p (numpy.ndarray): Predicted probabilities (between 0 and 1).
            clip (bool): Whether or not to clip predicted values see method clip.


        Returns:
            float: Cross-entropy loss value.
        """
        self.y,self.p = y,p
        if clip:self.clip()
        return (-self.y*numpy.log(self.p)).sum(axis=1).mean()
        
class MSE(Cost):
    r"""
    Mean Squared Error (MSE) Loss.

    This class computes the mean squared error loss between true labels (y) and predicted values (p).

    $$
    \displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}\left(y_{i}-{\hat {y_{i}}}\right)^{2}
    $$

    Methods:
        - compute(y: numpy.ndarray, p: numpy.ndarray) -> float:
            Computes the mean squared error loss.

        - pr() -> numpy.ndarray:
            Computes the derivative function values.

    Example:
        ```python
                >>> y_true = numpy.array([[2.0], [3.5], [5.0], [4.2]])
                >>> predicted_values = numpy.array([[1.8], [3.2], [4.8], [4.0]])
                >>> mse_loss = MSE()
                >>> loss_value = mse_loss.compute(y_true, predicted_values)
                >>> print(f"Mean Squared Error Loss: {loss_value:.4f}")
                Mean Squared Error Loss:  0.0525
                >>> derivative_values = mse_loss.pr()
                >>> print(f"Derivative Function Values: {derivative_values}")
                Derivative Function Values: [[-0.4]
                [-0.6]
                [-0.4]
                [-0.4]]
        ```
    """ 
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.ndarray:
        """
        Computes the derivative function values  with respet to p .

        Returns:
            numpy.ndarray: Derivative function values.
        """
        return -2*(self.y-self.p)
    def compute(self,y: numpy.ndarray,p: numpy.ndarray) -> float:
        """
        Computes the mean squared error loss.

        Args:
            y (numpy.ndarray): True labels (ground truth).
            p (numpy.ndarray): Predicted values.

        Returns:
            float: Mean squared error loss value.
        """
        self.y,self.p = y,p
        return ((self.y-self.p)**2).mean()