"""
    This modules provides genric templates for other modules

    - `Define` - generic object
    - `Architecture` - Architecture super object
    - `Layer` - layer super object
	- `Neurons` - neurons super object
	- `Cost` - cost super object
	- `Metrics` - weight super object

"""
from .utils import unfold,numpy
from .db import DBmanager,get_instance,update_instance,tables

class Define(DBmanager):

    __store = False

    def __repr__(self) -> str:
        """
        Returns the name of the class.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    @property
    def id(self) -> dict:
        return self._id 
    
    @id.setter
    def id(self,loc) -> None:
        """
        Sets id property for the class
        Instantiate and stores sqlalchemy table of self
        
        """
        loc = unfold(loc)
        self._id = {'id':id(self),f'{str(self)}_id':id(self),'name':repr(self),**loc}
        if Define._Define__store:
            if not hasattr(self,'table'):
                self.table = get_instance(self)
                self.add_table(self.table)
            else:
                update_instance(self)
                self.commit()

    def __getitem__(self,ix) -> any:
        return self.id[ix]
    def __setitem__(self,ix,val) -> None:
        self.id[ix] = val
    
    @property
    def get(self) -> dict.get:
        return self.id.get


    def __add__(self,loc:dict) -> None:
        """
        Triggers the id property

        Args:
            loc (dict) : dictionary of properties


        """
        self.id = loc
        self.c = 0
        class func:
            def __init__(self,_):...
        self.init_method = self.get('Layer_init_method',func)
        self.func = self.get('func',func)
        self.func = self.func(self.id)
        self['steps'] = self.get('steps',[])
        parent ={ f'{str(self)}_id': self['id']}
        for step in self : 
            step.id = {**step.id,**parent}

    def __iter__(self) -> object: return self
 
    def __len__(self) -> int:
        return len(self['steps'])
        
    
    def __next__(self) -> any:
        if self.c<len(self):
            self.c += 1
            return self['steps'][self.c-1]
        self.c = 0
        raise StopIteration   
    def commit(self) -> None:
        if Define._Define__store:
            DBmanager.session.commit()


    def predict(self,X:numpy.ndarray) -> numpy.ndarray:
        """
        Implements forward prediction of input feature matrix X of size n,k
        Passes outputs from input layer to output layer

        Args:
            X (numpy.ndarray) : input features matrix

        Returns:
            numpy.ndarray of output layer predictions


        """
        self.out = X
        for step in self:
            self.out = step.func.compute(self.out)
        return self.out

    def update(self,Δ:numpy.ndarray) -> numpy.ndarray :
        """
        Implement backpropagation of cost gradient to all layers
        Passes gradients backward

        Args:
            Δ (numpy.ndarray) : array of gradient from next step

        Returns:
            numpy array of input layer gradient
        
        """
        for step in self['steps'][::-1]:
            Δ = step.func.grad(Δ)
        return Δ

    def compute_store(self) -> None:
        """
        Generic method that computes item and stores it to sqlalchemy session

        """
        value = self.compute(self.y,self.p)
        if Define._Define__store:
            self.commit()
            del(self.table)
        self + {**self.id,**locals()}
        return value
    
    def updateW(self) -> None:
        """
        Updates sqlalchemy tables containing weights

        """
        for obj in Neurons.with_weights:
            for i,r in enumerate(obj.Wtables):
                for j,table in enumerate(r):
                    setattr(table,'value',obj.W[i,j])

class Layer(Define):
    """
    Model for layer functions 
    see :func:`~layer.fullyconnected`

    """
    def __str__(self) -> str:
        return 'Layer'  
 
class Neurons(Define):
    """
    Model for activation functions 
    see :func:`~activation.Softmax`

    """
    with_weights = []

    def instantiateW(self) -> None:
        """
        Instantiate weight tables
        """
        if Define._Define__store:
            table,cols = tables['Weight']
            self.Wtables = []
            for i,r in enumerate(self.W):
                instances = []
                for j,e in enumerate(r):
                    instances += [

                        table(Weight_id=f'{i}_{j}',
                            value=e,
                            Neurons_id=self['id']
                            )
                    ]
                self.Wtables += [instances]
                instances = []
            Neurons.with_weights += [self]


    def storeW(self):
        """
        Stores weights tables
        """
        if Define._Define__store:
            for row in self.Wtables:
                for table in row:
                    self.add_table(table)

    def __str__(self) -> str:
        return 'Neurons'
    
    def __sub__(self,Δ:numpy.ndarray) -> None:
        """
        Substracts Gradient to Weights

        """
        self.W -= Δ

    def n(self) -> int: 
        """
        Returns sample size for current features matrix
        """
        return self.X.shape[0]


    def grad(self,Δ:numpy.ndarray) -> numpy.ndarray:
        """
        Computes gradient for previous step

        Args:
            Δ (numpy.ndarray) : gradient from next step
        
        Returns:
            numpy.ndarray of gradient for previous step
        
        """
        self.Δ = self.pr()*Δ
        return self.Δ  

class Cost(Define):
    """
    Model for Cost functions 
    see :func:`~cost.binaryCrossEntropy`

    """
    def clip(self):
        """
        Applies numpy.clip function described bellow to the predicted probabilities
        It constrains values between [ε,1-ε] where ε=1e-7

        clip(a, a_min, a_max, out=None, **kwargs)
        Clip (limit) the values in an array.

        Given an interval, values outside the interval are clipped to
        the interval edges.  For example, if an interval of ``[0, 1]``
        is specified, values smaller than 0 become 0, and values larger
        than 1 become 1.

        Equivalent to but faster than ``np.minimum(a_max, np.maximum(a, a_min))``.

        No check is performed to ensure ``a_min < a_max``.

        Parameters
        ----------
        a : array_like
            Array containing elements to clip.
        a_min, a_max : array_like or None
            Minimum and maximum value. If ``None``, clipping is not performed on
            the corresponding edge. Only one of `a_min` and `a_max` may be
            ``None``. Both are broadcast against `a`.
        out : ndarray, optional
            The results will be placed in this array. It may be the input
            array for in-place clipping.  `out` must be of the right shape
            to hold the output.  Its type is preserved.
        **kwargs
            For other keyword-only arguments, see the
            :ref:`ufunc docs <ufuncs.kwargs>`.

            .. versionadded:: 1.17.0

        Returns
        -------
        clipped_array : ndarray
            An array with the elements of `a`, but where values
            < `a_min` are replaced with `a_min`, and those > `a_max`
            with `a_max`.

        See Also
        --------
        :ref:`ufuncs-output-type`

        Notes
        -----
        When `a_min` is greater than `a_max`, `clip` returns an
        array in which all values are equal to `a_max`,
        as shown in the second example.

        Examples
        --------
        ```python
                >>> a = np.arange(10)
                >>> a
                array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                >>> np.clip(a, 1, 8)
                array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
                >>> np.clip(a, 8, 1)
                array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                >>> np.clip(a, 3, 6, out=a)
                array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
                >>> a
                array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
                >>> a = np.arange(10)
                >>> a
                array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                >>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
                array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
        ```
        """
        ε = 1e-7
        self.p =self.p.clip(ε,1-ε)

    def __str__(self) -> str:
        return 'Cost'
    
class Metrics(Define):
    """
    Model for Metrics functions 
    see :func:`~metrics.accuracy`

    """
    def __str__(self) -> str:
        return 'Metrics'
    
class Architecture(Define):
    """
    Model for Architecture functions 
    see :func:`~architecture.Sequential`

    """
    def __str__(self) -> str:
        return 'Architecture'