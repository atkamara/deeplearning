"""
    This module provides neural network architectures
    Currently available are

    - `Sequential` - Sequential linear net architecture

"""
from .model import Architecture,Layer,Cost,Metrics,Define


class Sequential(Architecture):
    def __init__(self,steps: list[Layer],cost: Cost,store : bool=False) -> None:
        """
        Initialize a Sequential class.

        Args:
            steps (List[Layer]): A list of Layer objects representing the steps.
            cost (Cost): A Cost object for computing cost information.
            store (bool): If True disables identification and storage

        Example:
        ```python
                layer1 = Fullyconnected(2,50,init_funcs.zeros)
                layer2 = Activation(activation.LeakyReLU)
                my_cost = binaryCrossEntropy
                my_instance = Sequential(steps=[layer1, layer2], cost=my_cost)
        ```
        """
        Define._Define__store = store
        self + locals()
        self['cost'] = self['cost'](self['id'])
        self.commit()