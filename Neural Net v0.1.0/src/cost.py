"""
    This modules provides classes for several types of cost functions

    - `binaryCrossEntropy` 
    - `CrossEntropy` 
    - `MSE`

"""
from .utils import numpy
from .model import Cost


class binaryCrossEntropy(Cost):
    """
    Binary Cross-Entropy Loss.

    This class computes the binary cross-entropy loss between true labels (y) and predicted probabilities (p).

    Methods:
        - compute(y: numpy.array, p: numpy.array) -> float:
            Computes the binary cross-entropy loss.

        - pr(y: numpy.array, p: numpy.array) -> numpy.array:
            Computes the derivative function values.

    Example:
        >>> y_true = numpy.array([[0], [1], [1], [0]])
        >>> predicted_probs = numpy.array([[0.2], [0.8], [0.6], [0.3]])
        >>> bce_loss = binaryCrossEntropy()
        >>> loss_value = bce_loss.compute(y_true, predicted_probs)
        >>> print(f"Binary Cross-Entropy Loss: {loss_value:.4f}")
        Binary Cross-Entropy Loss: 0.3284
        >>> derivative_values = bce_loss.pr()
        >>> print(f"Derivative Function Values: {derivative_values}")
        Derivative Function Values: [ 1.25       -1.25       -1.66666667  1.42857143]
    """    
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.array:
        """
        Computes the derivative function values  with respet to p.

        Returns:
            numpy.array: Derivative function values.
        """
        return -(self.y/self.p - (1-self.y)/(1-self.p))
    def compute(self,y: numpy.array,p: numpy.array) -> float:
        """
        Computes the binary cross-entropy loss.

        Args:
            y (numpy.array): True labels (0 or 1).
            p (numpy.array): Predicted probabilities (between 0 and 1).

        Returns:
            float: Binary cross-entropy loss value.
        """
        self.y,self.p = y,p
        self.clip()
        return -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
   
class CrossEntropy(Cost):
    """
    Cross-Entropy Loss.

    This class computes the cross-entropy loss between true labels (y) and predicted probabilities (p).

    Methods:
        - compute(y: numpy.array, p: numpy.array) -> float:
            Computes the cross-entropy loss.

        - pr() -> numpy.array:
            Computes the derivative function values.

    Example:
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
    """
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.array:
        """
        Computes the derivative function values  with respet to p .

        Returns:
            numpy.array: Derivative function values.
        """
        return -(self.y/self.p - (1-self.y)/(1-self.p)) 
    def compute(self,y: numpy.array,p: numpy.array) -> float:
        """
        Computes the Cross-entropy loss.

        Args:
            y (numpy.array): True labels (0 or 1).
            p (numpy.array): Predicted probabilities (between 0 and 1).

        Returns:
            float: Cross-entropy loss value.
        """
        self.y,self.p = y,p
        self.clip()
        return -(self.y*numpy.log(self.p) + (1-self.y)*numpy.log(1-self.p)).mean()
        
class MSE(Cost):
    """
    Mean Squared Error (MSE) Loss.

    This class computes the mean squared error loss between true labels (y) and predicted values (p).

    Methods:
        - compute(y: numpy.array, p: numpy.array) -> float:
            Computes the mean squared error loss.

        - pr() -> numpy.array:
            Computes the derivative function values.

    Example:
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
    """ 
    def __init__(self,Architecture_id=None) -> None:
        self + locals()
    def pr(self) -> numpy.array:
        """
        Computes the derivative function values  with respet to p .

        Returns:
            numpy.array: Derivative function values.
        """
        return -2*(self.y-self.p)
    def compute(self,y: numpy.array,p: numpy.array) -> float:
        """
        Computes the mean squared error loss.

        Args:
            y (numpy.array): True labels (ground truth).
            p (numpy.array): Predicted values.

        Returns:
            float: Mean squared error loss value.
        """
        self.y,self.p = y,p
        return ((self.y-self.p)**2).mean()
