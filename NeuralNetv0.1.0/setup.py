from setuptools import setup

description ="""
 Neural Network v0.1.0
=====================
<a href="https://ibb.co/mGcm59P"><img src="https://i.ibb.co/dr5S4PH/nn.png" alt="nn" border="0"></a>
# [Documentation](https://atkamara.github.io/NeuralNetv0.1.0/)
# Quick start

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
""" 

setup(
    name="neural_net_numpy",
    version="0.1.0",
    author="Abdourahmane Tintou KAMARA",
    author_email="abdourahmane29@outlook.com",
    packages=["src"],
    long_description=description, 
    long_description_content_type="text/markdown",
    url="https://github.com/atkamara/deeplearning/tree/main/Neural%20Net%20v0.1.0", 
    license='MIT', 
    python_requires='>=3.8', 
    install_requires=["tqdm>=4.66.2","numpy>=1.26.4","SQLAlchemy>=2.0.27","pandas>=2.2.1"]
)