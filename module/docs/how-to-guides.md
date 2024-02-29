Building neural networks
========================

# Creating a Sequential Neural Network

## 1. Import all modules from neural_net package

```python
from neural_net import *
import numpy
```


## 2. Define Your Model

```python
ann_sigmoid = architecture.Sequential(
     [

       layers.Fullyconnected(n_in=2,n_out=50,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.σ),
       layers.Fullyconnected(n_in=50,n_out=1,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.σ),


    ],
    cost = cost.binaryCrossEntropy
)
```

## 3. Import or create your training dataset

```python
n,k = 5000,2
X = numpy.random.uniform(-100,100,size=(n,k))
y =( (X[:, 0]**2 + X[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0
```

## 4. Train your model

```python
ann_sigmoid.train(X,y,metrics=metrics.accuracy)
```

## 5. Make predictions

```python
ann_sigmoid.predict(X)
```