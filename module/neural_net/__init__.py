"""
    This packages provides modules for computing neural networks

    Modules exported by this package:

    - `activation`: provides classes for several types of activation functions
    - `layers`: This modules provides Layer classes
    - `architecture`: This modules provides neural network architectures
    - `init_funcs`: This modules provides initialization functions
    - `cost`: This modules provides classes for several types of cost functions
    - `metrics`: This modules provides metrics classes
    - `db`: This modules provides sqlalchemy orm tables and utility objects
    - `pipeline`: This modules provides functions for data preparation
"""
from . import layers
from . import activation
from . import architecture
from . import init_funcs
from . import cost
from . import metrics
from . import db
from . import pipeline
from . import model
from . import utils

__version__ = "0.1.0"
__author__ = 'Abdourahmane Tintou Kamara'
