Neural-Net-Numpy(NNN)
========================

# Creating a Sequential Neural Network

## 0. Install neural-net-numpy

```bash
$ pip install -i https://test.pypi.org/simple/ neural-net-numpy
```

## 1. Import modules from neural_net package

```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros,XavierHe
from neural_net.activation import σ,Softmax,LeakyReLU,Tanh,ELU,ReLU
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.metrics import accuracy
from neural_net.pipeline import onehot,scaler,shuffle,Batch
from neural_net.utils import IrisDatasetDownloader
```


## 2. Define Your Model

```python
NNN = Sequential(
        [
        Fullyconnected(2,50,XavierHe("Uniform","ReLU").init_func),
        Activation(LeakyReLU),     
        Fullyconnected(50,1,XavierHe("Uniform","Sigmoid").init_func),
        Activation(σ)
        ],
    BinaryCrossEntropy
    )
```

## 3. Import or create your training dataset

```python
import numpy

n,k = 5000,2
X = numpy.random.uniform(-100,100,size=(n,k))
y =( (X[:, 0]**2 + X[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0
```

## 4. Train your model

```python
NNN.train(scaler(X),y,α=α,epochs=n_epoch,metrics=accuracy)
```

## 5. Make predictions

```python
NNN.predict(X)
```