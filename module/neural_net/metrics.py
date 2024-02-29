"""
    This modules provides metrics classes

    - `accuracy` 

"""

from .utils import numpy
from .model import Metrics

class accuracy(Metrics):
    """
    Calculates the accuracy metric for binary or multiclass classification tasks.

    Args:
        threshold (float, optional): Threshold value for binary classification. Defaults to 0.5.

    Attributes:
        threshold (float): The threshold value used for binary classification.

    Methods:
        compute(y, p):
            Computes the accuracy score based on true labels (y) and predicted probabilities (p).

    Example:
    ```python
            >>> acc = accuracy(threshold=0.6)
            >>> y_true = numpy.array([[1], [0], [1], [0]])
            >>> y_pred = numpy.array([[0.8], [0.3], [0.9], [0.5]])
            >>> val = acc.compute(y_true, y_pred)
            >>> print(f"Accuracy: {val:.4f}")
            Accuracy: 1.0000
            >>> y_true_multiclass = numpy.array([[0, 0, 1],
            ...        [0, 1, 0],
            ...        [1, 0, 0],
            ...        [0, 0, 1],
            ...        [0, 1, 0],
            ...        [1, 0, 0],
            ...        [0, 1, 0],
            ...        [0, 0, 1]])
            >>> y_pred_multiclass = numpy.array([
            ...     [0.1, 0.2, 0.7],  # Predicted probabilities for class 0
            ...     [0.6, 0.3, 0.1],  # Predicted probabilities for class 1
            ...     [0.8, 0.1, 0.1],  # Predicted probabilities for class 2
            ...     [0.2, 0.3, 0.5],
            ...     [0.4, 0.4, 0.2],
            ...     [0.7, 0.2, 0.1],
            ...     [0.3, 0.4, 0.3],
            ...     [0.1, 0.2, 0.7]
            ... ])
            >>> model_multiclass = accuracy(threshold=0.5)
            >>> acc_multiclass = model_multiclass.compute(y_true_multiclass, y_pred_multiclass)
            >>> print(f"Accuracy (multiclass): {acc_multiclass:.4f}")
            Accuracy (multiclass): 0.7500
    ```

    """
    def __init__(self,threshold = .5) -> None:
        """
        Initializes the accuracy metric.

        Args:
            threshold (float, optional): Threshold value for binary classification. Defaults to 0.5.
        """
        self.threshold = threshold
    def compute(self,y: numpy.array,p: numpy.array) -> float:
        """
        Computes the accuracy of predictions.

        Args:
            y (numpy.array): True labels (ground truth).
            p (numpy.array): Predicted values.

        Returns:
            float: accuracy value.
        """
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