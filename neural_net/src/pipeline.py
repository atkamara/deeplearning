from .utils import numpy,pandas

def get_ix(size: int,obs: int) -> list[slice]:  
    """
    Create batch slices for a given sample size and batch size.

    Args:
        obs (int): Total number of samples in the dataset.
        size (int): Size of each batch.

    Returns:
        list[slice]: A list of slice objects representing batch indices.

    Example:
        >>> obs = 70  # Total samples
        >>> batch_size = 20
        >>> batch_slices = get_ix(obs, batch_size)
        >>> batch_slices
        [slice(0, 20, None), slice(20, 40, None), slice(40, 60, None), slice(60, 70, None)]
    """
    batchix = list(range(0,obs,size))
    if batchix[-1]<obs : batchix.append(obs)
    batchix = [slice(low,high) for low,high in zip(batchix,batchix[1:])]
    return batchix

def shuffle(X: numpy.array,y: numpy.array) -> tuple[numpy.array,numpy.array]:
    """
    shuffle features and tagert variable numpy arrays X and y using pandas.sample method.

    Args:
        X (numpy.array): Matrix of training features with shape (n, k), where n is the number of samples
                        and k is the number of features.
        y (numpy.array): Target variable with shape (n, 1).

    Returns:
        Tuple[numpy.array, numpy.array]: Shuffled X and y arrays.

    Example:
        >>> n,k = 5000,2
        >>> X_train = numpy.random.uniform(-100,100,size=(n,k))
        >>> y_train =( (X_train[:, 0]**2 + X_train[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0 
        >>> shuffled_X, shuffled_y = shuffle(X_train, y_train)
        # Now shuffled_X and shuffled_y contain randomly shuffled samples.
    """
    X = pandas.DataFrame(X).sample(frac=1)
    
    y = pandas.DataFrame(y).loc[X.index]

    return X.values,y.values

class Batch:

    def __init__(self,size: int, obs: int, X: callable, y :callable) -> None:
        """
        Initialize a Batch object.      
        Args:
            size (int): Size of each batch.
            obs (int): Total sample size.
            X (numpy.ndarray): function providing access to Numpy array containing features.
            y (numpy.ndarray): function providing access to Numpy array containing target variable.     
        Returns:
            None        
        Example:
            >>> def get_X():
            ...     return numpy.array([[1, 2], [3, 4], [5, 6]])
            >>> def get_y():
            ...     return numpy.array([[0], [1], [0]])
            >>> batch_size = 2
            >>> total_samples = len(X)
            >>> batch = Batch(size=batch_size, obs=total_samples, X=get_X, y=get_y)
            >>> for X_batch, y_batch in batch:
            ...     print(f"Features: {X_batch}, Target: {y_batch}")
            Features: [[1 2]
            [3 4]], Target: [[0]
            [1]]
            Features: [[5 6]], Target: [[0]]
        """
        self.size = size
        self.obs = obs
        self.X = X
        self.y = y
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

def onehot(y:numpy.array) -> numpy.array: 
    """
    One-hot encodes a categorical target variable.

    Args:
        y (numpy.array): Numpy array containing the categorical target variable.

    Returns:
        numpy.array: One-hot encoded representation of the target variable.

    Example:
        >>> y = numpy.array([[0],[ 1], [2], [1], [0]])
        >>> onehot_encoded = onehot(y)
        >>> print(onehot_encoded)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]
         [0. 1. 0.]
         [1. 0. 0.]]
    """
    return (y==numpy.unique(y))+0




def scaler(X: numpy.array) -> numpy.array:
    """
    Custom scaler function for centering and standardizing features.

    Args:
        X (numpy.array): Input numpy array containing features.

    Returns:
        numpy.array: Scaled version of the input array.

    Example:
        >>> X = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> scaled_X = scaler(X)
        >>> print(scaled_X)
        [[-1.22474487 -1.22474487]
         [ 0.          0.        ]
         [ 1.22474487  1.22474487]]
    """
    return (X-X.mean(axis=0))/X.std(axis=0)