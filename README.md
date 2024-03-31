Neural-Net-Numpy(nnn)
========================

<a href="https://ibb.co/mGcm59P"><img src="https://i.ibb.co/dr5S4PH/nn.png" alt="nn" border="0"></a>
# Check [Documentation](https://atkamara.github.io/neural-net-numpy/) for more
# Quick start


## Step 0. Install neural-net-numpy

```bash
$ pip install -i https://test.pypi.org/simple/ neural-net-numpy
```

## Step 1. Import modules from neural_net package

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


## Step 2. Define Your Model

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

## Step 3. Import or create your training dataset

```python
import numpy

n,k = 5000,2
X = numpy.random.uniform(-100,100,size=(n,k))
y =( (X[:, 0]**2 + X[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0
```

## Step 4. Train your model

```python
NNN.train(scaler(X),y,metrics=accuracy)
```

## Step 5. Make predictions

```python
NNN.predict(X)
```
