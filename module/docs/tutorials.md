# Tutorials


### Install  NNN library from TestPypi repo [link here](https://test.pypi.org/project/neural-net-numpy)

![png](static/1.png)

### Check python version (requirement python >=3.8)


```bash
$ python --version
```

    Python 3.11.7



```bash
$ pip install --upgrade pip
$ pip install -i https://test.pypi.org/simple/ neural-net-numpy
```

    Requirement already satisfied: pip in /home/analyst/dlenv/lib/python3.11/site-packages (24.0)
    Looking in indexes: https://test.pypi.org/simple/
    Collecting neural-net-numpy==0.1.4
      Downloading https://test-files.pythonhosted.org/packages/48/77/4d5e4d9de3f9bd758dd510a2d9a3dfb0566f3c90dcd8e40d81e3af815ef4/neural_net_numpy-0.1.4-py3-none-any.whl.metadata (1.8 kB)
    Requirement already satisfied: tqdm>=4.66.2 in /home/analyst/dlenv/lib/python3.11/site-packages (from neural-net-numpy==0.1.4) (4.66.2)
    Requirement already satisfied: numpy>=1.26.4 in /home/analyst/dlenv/lib/python3.11/site-packages (from neural-net-numpy==0.1.4) (1.26.4)
    Requirement already satisfied: SQLAlchemy>=2.0.27 in /home/analyst/dlenv/lib/python3.11/site-packages (from neural-net-numpy==0.1.4) (2.0.27)
    Requirement already satisfied: pandas>=2.2.0 in /home/analyst/dlenv/lib/python3.11/site-packages (from neural-net-numpy==0.1.4) (2.2.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/analyst/dlenv/lib/python3.11/site-packages (from pandas>=2.2.0->neural-net-numpy==0.1.4) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /home/analyst/dlenv/lib/python3.11/site-packages (from pandas>=2.2.0->neural-net-numpy==0.1.4) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /home/analyst/dlenv/lib/python3.11/site-packages (from pandas>=2.2.0->neural-net-numpy==0.1.4) (2024.1)
    Requirement already satisfied: typing-extensions>=4.6.0 in /home/analyst/dlenv/lib/python3.11/site-packages (from SQLAlchemy>=2.0.27->neural-net-numpy==0.1.4) (4.9.0)
    Requirement already satisfied: greenlet!=0.4.17 in /home/analyst/dlenv/lib/python3.11/site-packages (from SQLAlchemy>=2.0.27->neural-net-numpy==0.1.4) (3.0.3)
    Requirement already satisfied: six>=1.5 in /home/analyst/dlenv/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas>=2.2.0->neural-net-numpy==0.1.4) (1.16.0)
    Downloading https://test-files.pythonhosted.org/packages/48/77/4d5e4d9de3f9bd758dd510a2d9a3dfb0566f3c90dcd8e40d81e3af815ef4/neural_net_numpy-0.1.4-py3-none-any.whl (16 kB)
    Installing collected packages: neural-net-numpy
    Successfully installed neural-net-numpy-0.1.4


### Check install


```bash
$ python -c "import neural_net;print(neural_net.__version__)"
```

    0.1.0


## Import data science libraries

* ![png](static/f127ad61-800d-4e56-9961-99570d9fdf91.png){ width=100 height=70 }
* ![png](static/0a23e67b-5f88-4261-8727-7c03767d263a.png){ width=100 height=70}
* ![png](static/01e8788e-d3af-4f27-9830-336578c49a00.png){ width=100 height=70}


```python
import numpy,pandas
import matplotlib.pyplot as plt
```

## Activation functions


```python
z = numpy.linspace(-6,6,1000+1)
```

### Sigmoid and Tanh

#### Function values


```python
from neural_net.activation import σ,Tanh
```


```python
sigmoid = σ()
tanh    = Tanh()
```


```python
sigmoid.compute(z)
```




    array([0.00247262, 0.0025024 , 0.00253253, ..., 0.99746747, 0.9974976 ,
           0.99752738])




```python
tanh.compute(z)
```




    array([-1.99997542, -1.99997483, -1.99997421, ...,  1.99997421,
            1.99997483,  1.99997542])




```python
sigmoid.preds
```




    array([0.00247262, 0.0025024 , 0.00253253, ..., 0.99746747, 0.9974976 ,
           0.99752738])




```python
tanh.preds
```




    array([-1.99997542, -1.99997483, -1.99997421, ...,  1.99997421,
            1.99997483,  1.99997542])




```python
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

ax1.plot(z,sigmoid.preds,label=r'$\sigma=\frac{1}{1+e^{-z}}$')
ax2.plot(z,tanh.preds,label='$2\sigma(2z) - 1$')

ax1.hlines(y=0.5,xmin=-5,xmax=5,color='green',label=r'$y=\frac{1}{2}$',linestyle='--')
ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')


ax2.hlines(y=0,xmin=-5,xmax=5,color='red',label=r'$y=0$',linestyle='--')

ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_xlim(-5,5)

ax1.legend()
ax2.legend()

ax1.set_title('Sigmoid')
ax2.set_title('Hyperbolic Tangent Function')

ax1.grid()
ax2.grid()
```


    
![png](static/output_18_0.png)
    


#### Derivatives


```python
sigmoidpr = sigmoid.pr()
```


```python
tanhpr = tanh.pr()
```


```python
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

ax1.plot(z,sigmoidpr,label=r'$\sigma(1-\sigma)$')
ax2.plot(z,tanhpr,label='$1 - tanh^2$')

ax1.hlines(y=0.25,xmin=65,xmax=5,color='green',linestyle='--')
ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')


ax2.hlines(y=0,xmin=-5,xmax=5,color='red',linestyle='--')

ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_xlim(-5,5)

ax1.legend()
ax2.legend()

ax1.set_title('Sigmoid Derivative')
ax2.set_title('Hyperbolic Tangent Function Derivative')

ax1.grid()
ax2.grid()
```


    
![png](static/output_22_0.png)
    


### Rectified Linear Unit (ReLU)

#### Function values
$$
    \mathrm{\mathit{H}}(z) = \begin{cases}
        z & \text{if } z \geq 0  \\ % & is your "\tab"
        0 & \text{if } z < 0
    \end{cases}
    $$


```python
from neural_net.activation import ReLU
```


```python
relu = ReLU()
```


```python
relupred = relu.compute(z)
relupred
```




    array([0.   , 0.   , 0.   , ..., 5.976, 5.988, 6.   ])




```python
fig,ax1 = plt.subplots(nrows=1,ncols=1,figsize=(7.5,6))

ax1.plot(z,relupred,label=r'ReLU')

ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')



ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax1.legend()

ax1.set_title('Rectified Linear Unit')

ax1.grid()
```


    
![png](static/output_28_0.png)
    


#### Derivative

$$
\mathrm{\mathit{H}}(z) = \begin{cases}
1 & \text{if } z \geq 0  \\
0 & \text{if } z < 0
\end{cases}
$$


```python
relupr = relu.pr()
relupr
```




    array([0, 0, 0, ..., 1, 1, 1])




```python
fig,ax1 = plt.subplots(nrows=1,ncols=1,figsize=(7.5,6))

ax1.plot(z,relupr,label=r'ReLU')

ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')



ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)

ax1.legend()

ax1.set_title('Rectified Linear Unit Derivative')

ax1.grid()
```


    
![png](static/output_31_0.png)
    


### Non Saturating activations

#### Function values
- Leaky Rectified Linear Unit(Leaky ReLU)
$$
\mathrm{\mathit{H}}(z) = \begin{cases}
 z & \text{if }  z \geq 0  \\ 
 \alpha z & \text{if } z < 0
\end{cases}
$$

- Exponential Linear Unit(ELU)
$$
\mathrm{\mathit{H}}(z) = \begin{cases}
    z & \text{if } z \geq 0  \\ 
    \alpha (e^{z} - 1) & \text{if } z < 0
\end{cases}
$$


```python
from neural_net.activation import ELU,LeakyReLU
```


```python
elu = ELU(α=1)
leakyrelu = LeakyReLU(α=.1)
```


```python
elupred,leakyrelupred = elu.compute(z),leakyrelu.compute(z)
```


```python
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

ax1.plot(z,elupred,label=r'$\alpha=1$')
ax2.plot(z,leakyrelupred,label=r'$\alpha=0.1$')

ax1.hlines(y=0,xmin=-5,xmax=5,color='green',linestyle='--')
ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')


ax2.hlines(y=0,xmin=-5,xmax=5,color='red',linestyle='--')

ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_xlim(-5,5)
ax1.legend()
ax2.legend()

ax1.set_title('Exponential Linear Unit')
ax2.set_title('Leaky Rectified Linear Unit')

ax1.grid()
ax2.grid()
```


    
![png](static/output_37_0.png)
    


#### Derivatives
- ELU
$$
\mathrm{\mathit{H}}'(z) = \begin{cases}
 1 & \text{if } z \geq 0  \\ % &
  \mathrm{\mathit{H}}(z) + \alpha & \text{if } z < 0
  \end{cases}
$$ 
- Leaky ReLU
$$
\mathrm{\mathit{H}}'(z) = \begin{cases}
 1 & \text{if } z \geq 0  \\ % &
  \alpha & \text{if } z < 0
  \end{cases}
$$


```python
elupr,leakypr = elu.pr(),leakyrelu.pr()
```


```python
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

ax1.plot(z,elupr,label=r'$\alpha=1$')
ax2.plot(z,leakypr,label=r'$\alpha=0.1$')

ax1.hlines(y=0,xmin=-5,xmax=5,color='green',linestyle='--')
ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')


ax2.hlines(y=0,xmin=-5,xmax=5,color='red',linestyle='--')

ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_xlim(-5,5)
ax1.legend()
ax2.legend()

ax1.set_title('Exponential Linear Unit Derivative')
ax2.set_title('Leaky Rectified Linear Unit Derivative')

ax1.grid()
ax2.grid()
```


    
![png](static/output_40_0.png)
    


### All common activation function and their derivatives


```python
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

ax1.plot(z,elu.preds,label=r'$ELU(\alpha=1)$')
ax1.plot(z,leakyrelu.preds,label=r'$Leaky\ ReLU(\alpha=0.1)$')
ax1.plot(z,relu.preds,label=r'ReLU')
ax1.plot(z,sigmoid.preds,label=r'$\sigma=\frac{1}{1+e^{-z}}$')
ax1.plot(z,tanh.preds,label='$tanh=2\sigma(2z) - 1$')


ax2.plot(z,elupr,label=r'$ELU(\alpha=1)$')
ax2.plot(z,leakypr,label=r'$Leaky\ ReLU(\alpha=0.1$)')
ax2.plot(z,relupr,label=r'ReLU')
ax2.plot(z,sigmoidpr,label=r"$\sigma'=\sigma(1-\sigma)$")
ax2.plot(z,tanhpr,label='$tanh=1 - tanh^2$')

ax1.hlines(y=0,xmin=-5,xmax=5,color='green',linestyle='--')
ax1.vlines(x=0,ymin=-5,ymax=5,color='brown',linestyle='--')


ax2.hlines(y=0,xmin=-5,xmax=5,color='red',linestyle='--')

ax1.set_ylim(-5,5)
ax1.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax2.set_xlim(-5,5)
ax1.legend()
ax2.legend()

ax1.set_title('Common activation functions')
ax2.set_title('Derivatives')

ax1.grid()
ax2.grid()
```


    
![png](static/output_42_0.png)
    


## Initialization


```python
from neural_net.init_funcs import XavierHe, zeros
```

### Weights + Bias


```python
n_cols = 2
```

#### Zeros


```python
W = zeros(n_cols,1)
W
```




    array([[0.],
           [0.],
           [0.]])




```python
W = zeros(n_cols,1,biais=False)
W
```




    array([[0.],
           [0.]])



### Xavier and He


![png](static/xahe.png)


```python
initializer = XavierHe("Normal","Sigmoid").init_func
```


```python
initializer(n_cols,1)
```




    array([[ 1.08529782],
           [ 0.55315106],
           [-1.20349346]])




```python
initializer(n_cols,1,biais=False)
```




    array([[0.14914699],
           [0.68662846]])



#### Normal distribution


```python
n_cols = 1000
```


```python
xe_norm_sigmoid = XavierHe("Normal","Sigmoid").init_func(n_cols,1,biais=False)
xe_norm_tanh = XavierHe("Normal","Tanh").init_func(n_cols,1,biais=False)
xe_norm_relu = XavierHe("Normal","ReLU").init_func(n_cols,1,biais=False)

```


```python
xe_norm_sigmoid.shape
```




    (1000, 1)




```python
std,m = xe_norm_sigmoid.std(),xe_norm_sigmoid.mean()
std,m
```




    (0.04489498522857033, -0.0007231213204853488)




```python
(( xe_norm_sigmoid >= m-std) & (xe_norm_sigmoid<=m+std)).sum()
```




    697




```python
697/n_cols
```




    0.697




```python
(( xe_norm_sigmoid >= m-2*std) & (xe_norm_sigmoid<=m+2*std)).sum()/n_cols
```




    0.95




```python
(( xe_norm_sigmoid >= m-3*std) & (xe_norm_sigmoid<=m+3*std)).sum()/n_cols
```




    0.997




```python
xe_norm_tanh.std(),xe_norm_relu.std()
```




    (0.1711681672411125, 0.06410140383405528)




```python
plt.figure(figsize=(10,6))

plt.title('Xavier/He for normal distribution by activation type')
plt.hist(xe_norm_tanh,label=r'Tanh $std=4\sigma$',bins=30,alpha=.4)
plt.hist(xe_norm_relu,label=r'ReLU $std=\sqrt{2}\sigma$',bins=30,alpha=.5)
plt.hist(xe_norm_sigmoid,label=r'Sigmoid $std=\sigma=\sqrt{\frac{2}{n_{in}+n_{out}}}$',bins=30,alpha=.6)

plt.legend(loc='upper left')
plt.axis('off')
```




    (-0.726385463544662, 0.6217518101350396, 0.0, 111.3)




    
![png](static/output_64_1.png)
    


#### Uniform distribution


```python
xe_uni_sigmoid = XavierHe("Uniform","Sigmoid").init_func(n_cols,1,biais=False)
xe_uni_tanh = XavierHe("Uniform","Tanh").init_func(n_cols,1,biais=False)
xe_uni_relu = XavierHe("Uniform","ReLU").init_func(n_cols,1,biais=False)

```


```python
xe_uni_sigmoid.min()
```




    -0.07708092487365578




```python
xe_uni_sigmoid.max()
```




    0.0770798990273792




```python
plt.figure(figsize=(10,6))

plt.title('Xavier/He for uniform distribution by activation type')
plt.hist(xe_uni_tanh,label=r'Tanh $r=4m$',bins=25,alpha=.4)
plt.hist(xe_uni_relu,label=r'ReLU $r=\sqrt{2}m$',bins=25,alpha=.5)
plt.hist(xe_uni_sigmoid,label=r'Sigmoid $r=m=\sqrt{\frac{2}{n_{in}+n_{out}}}$',bins=25,alpha=.6)

plt.legend(loc='upper left')

```




    <matplotlib.legend.Legend at 0x7fac479fe350>




    
![png](static/output_69_1.png)
    


## Layers


```python
from neural_net.layers import Fullyconnected,Activation

```

### Linear Layer


```python
fc = Fullyconnected(n_in=2,n_out=1,init_method=zeros)
```


```python
repr(fc)
```




    'Fullyconnected'




```python
str(fc)
```




    'Layer'




```python
fc.id
```




    {'id': 140377909934032,
     'Layer_id': 140377909934032,
     'name': 'Fullyconnected',
     'self': Fullyconnected,
     'n_in': 2,
     'n_out': 1,
     'init_method': <function neural_net.init_funcs.zeros(n_in: int, n_out: int, biais: bool = True) -> <built-in function array>>,
     'func': neural_net.activation.Σ,
     'steps': []}




```python
fc['id']
```




    140377909934032




```python
fc.id['id']
```




    140377909934032




```python
fc.func
```




    Σ



### Linear activation


```python
linear_activation = fc.func
linear_activation
```




    Σ




```python
str(linear_activation)
```




    'Neurons'




```python
repr(linear_activation)
```




    'Σ'




```python
linear_activation.id
```




    {'id': 140377909866128,
     'Neurons_id': 140377909866128,
     'name': 'Σ',
     'self': Σ,
     'Layer_id': 140377909934032,
     'Layer_Layer_id': 140377909934032,
     'Layer_name': 'Fullyconnected',
     'Layer_self': Fullyconnected,
     'Layer_n_in': 2,
     'Layer_n_out': 1,
     'Layer_init_method': <function neural_net.init_funcs.zeros(n_in: int, n_out: int, biais: bool = True) -> <built-in function array>>,
     'Layer_func': neural_net.activation.Σ,
     'steps': []}




```python
linear_activation.W
```




    array([[0.],
           [0.],
           [0.]])




```python
linear_activation.W.shape
```




    (3, 1)



### activation layer


```python
sigmoid_activation = Activation(func=σ)
```


```python
sigmoid_activation
```




    Activation




```python
str(sigmoid_activation)
```




    'Layer'




```python
sigmoid_activation.id
```




    {'id': 140377909680656,
     'Layer_id': 140377909680656,
     'name': 'Activation',
     'self': Activation,
     'func': neural_net.activation.σ,
     'kargs': (),
     'steps': []}



## Architecture


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ
from neural_net.cost import BinaryCrossEntropy
```


```python
network = Sequential(
        [
        Fullyconnected(2,10,zeros),
        Activation(σ),
        Fullyconnected(10,1,zeros),
        Activation(σ)
        ]
    ,BinaryCrossEntropy)
```


```python
repr(network)
```




    'Sequential'




```python
str(network)
```




    'Architecture'




```python
network['steps']
```




    [Fullyconnected, Activation, Fullyconnected, Activation]




```python
network.id.keys()
```




    dict_keys(['id', 'Architecture_id', 'name', 'self', 'steps', 'cost', 'store'])




```python
network['id']
```




    140379602988880




```python
network.id['id']
```




    140379602988880



## Adding Database


```python
network = Sequential(
        [
        Fullyconnected(2,50,zeros),
        Activation(σ),
        Fullyconnected(50,2,zeros),
        Activation(σ)
        ]
    ,BinaryCrossEntropy,store=True)
```


```python
network.session
```




    <sqlalchemy.orm.session.Session at 0x7face42132d0>




```python
network.db_path
```




    'sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db'




```python
utc_ts = network.db_path.split('/')[-1][5:-3]
utc_ts
```




    '1709575905'




```python
import datetime
```


```python
datetime.datetime.fromtimestamp(int(utc_ts)).isoformat()
```




    '2024-03-04T18:11:45'




```python
db_folder = '/'.join(network.db_path.split('/')[3:-1])
db_folder
```




    '/home/analyst/notebooks/module/neural_net/run'




```python
%ls  $db_folder/*db|tail -n 3
```

    /home/analyst/notebooks/module/neural_net/run/model1709575470.db
    /home/analyst/notebooks/module/neural_net/run/model1709575647.db
    /home/analyst/notebooks/module/neural_net/run/model1709575905.db



```python
network.engines
```




    {'sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db': Engine(sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db)}




```python
network.engines.get(network.db_path)
```




    Engine(sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db)




```python
cursor = network.engines.get(network.db_path).connect()
```


```python
from sqlalchemy import text
```


```python

res = cursor.execute(text('''

        SELECT * 
        FROM
        sqlite_schema

'''))
pandas.DataFrame(res.fetchall())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>name</th>
      <th>tbl_name</th>
      <th>rootpage</th>
      <th>sql</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>table</td>
      <td>Architecture</td>
      <td>Architecture</td>
      <td>2</td>
      <td>CREATE TABLE "Architecture" (\n\tid INTEGER NO...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>table</td>
      <td>Layer</td>
      <td>Layer</td>
      <td>3</td>
      <td>CREATE TABLE "Layer" (\n\t"Architecture_id" IN...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>table</td>
      <td>Cost</td>
      <td>Cost</td>
      <td>4</td>
      <td>CREATE TABLE "Cost" (\n\t"Architecture_id" INT...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>table</td>
      <td>Neurons</td>
      <td>Neurons</td>
      <td>5</td>
      <td>CREATE TABLE "Neurons" (\n\t"Layer_id" INTEGER...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>table</td>
      <td>Weight</td>
      <td>Weight</td>
      <td>6</td>
      <td>CREATE TABLE "Weight" (\n\tvalue INTEGER, \n\t...</td>
    </tr>
  </tbody>
</table>
</div>



## Forward Feeding data to network

### Generating Linearly seperable data


```python
n,k = 300,2
X = numpy.random.uniform(-100,100,size=(n,k))
y = (X.sum(axis=1) < numpy.random.uniform(.3,.37,(len(X),))).reshape(-1,1)+0
plt.scatter(x=X[:,0],y=X[:,1],c=y)

```




    <matplotlib.collections.PathCollection at 0x7faca51e37d0>




    
![png](static/output_117_1.png)
    


### Looping over layers


```python
for layer in network:
    print(repr(layer))
```

    Fullyconnected
    Activation
    Fullyconnected
    Activation



```python
layer.func
```




    σ




```python
out = X
for layer in network:
    out = layer.func.compute(out)
```


```python
out
```




    array([[0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5]])




```python
out.shape
```




    (300, 2)



### Using predict method


```python
network.predict(X)
network.out.shape
```




    (300, 2)



## Exploring database


```python
%load_ext sql
%sql $network.db_path
```

### Architecture


```sql
%%sql

SELECT *
FROM Architecture
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Architecture_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>2024-03-04 18:11:45.895299</td>
            <td>2024-03-04 18:11:45.895302</td>
            <td>Sequential</td>
            <td>140379602670032</td>
        </tr>
        <tr>
            <td>2</td>
            <td>2024-03-04 18:17:26.316307</td>
            <td>2024-03-04 18:17:26.316315</td>
            <td>Sequential</td>
            <td>140379603024400</td>
        </tr>
    </tbody>
</table>




```python
network['id']
```




    140379603024400



### Costs


```sql
%%sql

SELECT *
FROM cost
WHERE Architecture_id = 140379603024400
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>Architecture_id</th>
            <th>value</th>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Cost_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>140379603024400</td>
            <td>None</td>
            <td>2</td>
            <td>2024-03-04 18:17:26.591265</td>
            <td>2024-03-04 18:17:26.591271</td>
            <td>BinaryCrossEntropy</td>
            <td>140379601731216</td>
        </tr>
    </tbody>
</table>



### Layers


```sql
%%sql   

SELECT * 
FROM layer 
WHERE Architecture_id=140379603024400
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>Architecture_id</th>
            <th>n_in</th>
            <th>n_out</th>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Layer_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>140379603024400</td>
            <td>2</td>
            <td>50</td>
            <td>5</td>
            <td>2024-03-04 18:17:26.327160</td>
            <td>2024-03-04 18:17:26.327167</td>
            <td>Fullyconnected</td>
            <td>140379603033872</td>
        </tr>
        <tr>
            <td>140379603024400</td>
            <td>None</td>
            <td>None</td>
            <td>6</td>
            <td>2024-03-04 18:17:26.327168</td>
            <td>2024-03-04 18:17:26.513963</td>
            <td>Activation</td>
            <td>140379602907280</td>
        </tr>
        <tr>
            <td>140379603024400</td>
            <td>50</td>
            <td>2</td>
            <td>7</td>
            <td>2024-03-04 18:17:26.327169</td>
            <td>2024-03-04 18:17:26.547988</td>
            <td>Fullyconnected</td>
            <td>140379601709072</td>
        </tr>
        <tr>
            <td>140379603024400</td>
            <td>None</td>
            <td>None</td>
            <td>8</td>
            <td>2024-03-04 18:17:26.327170</td>
            <td>2024-03-04 18:17:26.572231</td>
            <td>Activation</td>
            <td>140379601815760</td>
        </tr>
    </tbody>
</table>



### Neurons


```sql
%%sql   

SELECT * 
FROM neurons 
WHERE layer_id=140379603033872
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>Layer_id</th>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Neurons_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>140379603033872</td>
            <td>1</td>
            <td>2024-03-04 18:17:26.330924</td>
            <td>2024-03-04 18:17:26.330930</td>
            <td>Σ</td>
            <td>140379601518480</td>
        </tr>
    </tbody>
</table>



### Weights


```sql
%%sql

SELECT * 
FROM
weight WHERE neurons_id = 140379601518480
LIMIT 10
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>value</th>
            <th>Neurons_id</th>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Weight_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>1</td>
            <td>2024-03-04 18:17:26.341093</td>
            <td>2024-03-04 18:17:26.341100</td>
            <td>None</td>
            <td>0_0</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>2</td>
            <td>2024-03-04 18:17:26.341101</td>
            <td>2024-03-04 18:17:26.341102</td>
            <td>None</td>
            <td>0_1</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>3</td>
            <td>2024-03-04 18:17:26.341102</td>
            <td>2024-03-04 18:17:26.341103</td>
            <td>None</td>
            <td>0_2</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>4</td>
            <td>2024-03-04 18:17:26.341104</td>
            <td>2024-03-04 18:17:26.341104</td>
            <td>None</td>
            <td>0_3</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>5</td>
            <td>2024-03-04 18:17:26.341105</td>
            <td>2024-03-04 18:17:26.341105</td>
            <td>None</td>
            <td>0_4</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>6</td>
            <td>2024-03-04 18:17:26.341106</td>
            <td>2024-03-04 18:17:26.341106</td>
            <td>None</td>
            <td>0_5</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>7</td>
            <td>2024-03-04 18:17:26.341107</td>
            <td>2024-03-04 18:17:26.341107</td>
            <td>None</td>
            <td>0_6</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>8</td>
            <td>2024-03-04 18:17:26.341108</td>
            <td>2024-03-04 18:17:26.341108</td>
            <td>None</td>
            <td>0_7</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>9</td>
            <td>2024-03-04 18:17:26.341109</td>
            <td>2024-03-04 18:17:26.341109</td>
            <td>None</td>
            <td>0_8</td>
        </tr>
        <tr>
            <td>0</td>
            <td>140379601518480</td>
            <td>10</td>
            <td>2024-03-04 18:17:26.341110</td>
            <td>2024-03-04 18:17:26.341110</td>
            <td>None</td>
            <td>0_9</td>
        </tr>
    </tbody>
</table>




```sql
%%sql

SELECT count(*) n_neurons, AVG(value) mean_value
FROM
weight WHERE neurons_id = 140379601518480
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709575905.db
    Done.





<table>
    <thead>
        <tr>
            <th>n_neurons</th>
            <th>mean_value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>150</td>
            <td>0.0</td>
        </tr>
    </tbody>
</table>



## Predict Method


```python
network.predict(X)
```




    array([[0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5]])



## Cost functions


```python
from neural_net.cost import BinaryCrossEntropy, CrossEntropy, MSE
from neural_net.utils import make_circle_data,IrisDatasetDownloader,HouseDatasetDownloader,Pearson,Boostrap
import numpy
import matplotlib.pyplot as plt
```

### Binary Crossentropy
$$
\mathrm{\mathit{Binary\ Cross\ Entropy}}(p, y) = \begin{cases}
-\log(p) & \text{if } y = 1, \\
-\log(1-p) & \text{otherwise.}
\end{cases}
$$


```python
bcost = BinaryCrossEntropy()
```

#### Circles dataset


```python
centers = [(-50, 0), (20, 30)]
radii = [30, 35]
X, y = make_circle_data(centers, radii)
print(X.shape, y.shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=y)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
```

    (328, 2) (328, 1)





    (-100.0, 100.0)




    
![png](static/output_147_2.png)
    



```python
dum_classifier = numpy.random.random(len(y))
dum_classifier.shape
```




    (328,)




```python
bcost.compute(y,dum_classifier)
```




    0.9439193222155523




```python
round(bcost.compute(y,y))
```




    0



#### Properties

##### With clipped values( default clip=True)


```python
ps = numpy.linspace(0,1,1000).reshape(-1,1)
y1 = numpy.array([ [[bcost.compute(numpy.array([1]),p)],bcost.pr()] for p in ps ])
y0 = numpy.array([ [[bcost.compute(numpy.array([0]),p)],bcost.pr()] for p in ps ])
```


```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps,y1[:,0],label='estimated probabilities')
ax2.set_title('y=0')
ax2.plot(ps,y0[:,0],label='estimated probabilities')
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d6a6e3d90>




    
![png](static/output_154_1.png)
    


###### Derivaties


```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps,y1[:,1],label="$BCE'_{p}$")
ax2.set_title('y=0')
ax2.plot(ps,y0[:,1],label="$BCE'_{p}$")
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d68199a90>




    
![png](static/output_156_1.png)
    



```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps[10:-10,:],y1[10:-10,1],label="$BCE'_{p}$")
ax2.set_title('y=0')
ax2.plot(ps[10:-10,:],y0[10:-10:,1],label="$BCE'_{p}$")
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d680e5f10>




    
![png](static/output_157_1.png)
    


##### Without clippping


```python
ps = numpy.linspace(1e-9,1-1e-9,1000).reshape(-1,1)
y1 = numpy.array([ [[bcost.compute(numpy.array([1]),p,clip=False)],bcost.pr()] for p in ps ])
y0 = numpy.array([ [[bcost.compute(numpy.array([0]),p,clip=False)],bcost.pr()] for p in ps ])
```


```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps,y1[:,0],label='estimated probabilities')
ax2.set_title('y=0')
ax2.plot(ps,y0[:,0],label='estimated probabilities')
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d67d6a490>




    
![png](static/output_160_1.png)
    


###### Derivaties


```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps,y1[:,1],label="$BCE'_{p}$")
ax2.set_title('y=0')
ax2.plot(ps,y0[:,1],label="$BCE'_{p}$")
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d680bd010>




    
![png](static/output_162_1.png)
    



```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
ax1.set_title('y=1')
ax1.plot(ps[10:-10,:],y1[10:-10,1],label="$BCE'_{p}$")
ax2.set_title('y=0')
ax2.plot(ps[10:-10,:],y0[10:-10:,1],label="$BCE'_{p}$")
ax1.legend()
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d67b1f6d0>




    
![png](static/output_163_1.png)
    


### Cross Entropy


```python
ce = CrossEntropy()
```

#### Iris dataset


```python
iris = IrisDatasetDownloader()
iris.load_dataset()

```


```python
print(iris.description)
```

    
            1. Title: Iris Plants Database
                Updated Sept 21 by C.Blake - Added discrepency information
    
            2. Sources:
                (a) Creator: R.A. Fisher
                (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
                (c) Date: July, 1988
    
            3. Past Usage:
                - Publications: too many to mention!!!  Here are a few.
            1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
                Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions
                to Mathematical Statistics" (John Wiley, NY, 1950).
            2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
                (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
            3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
                Structure and Classification Rule for Recognition in Partially Exposed
                Environments".  IEEE Transactions on Pattern Analysis and Machine
                Intelligence, Vol. PAMI-2, No. 1, 67-71.
                -- Results:
                    -- very low misclassification rates (0% for the setosa class)
            4. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE 
                Transactions on Information Theory, May 1972, 431-433.
                -- Results:
                    -- very low misclassification rates again
            5. See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al's AUTOCLASS II
                conceptual clustering system finds 3 classes in the data.
    
            4. Relevant Information:
                --- This is perhaps the best known database to be found in the pattern
                    recognition literature.  Fisher's paper is a classic in the field
                    and is referenced frequently to this day.  (See Duda & Hart, for
                    example.)  The data set contains 3 classes of 50 instances each,
                    where each class refers to a type of iris plant.  One class is
                    linearly separable from the other 2; the latter are NOT linearly
                    separable from each other.
                --- Predicted attribute: class of iris plant.
                --- This is an exceedingly simple domain.
                --- This data differs from the data presented in Fishers article
                    (identified by Steve Chadwick,  spchadwick@espeedaz.net )
                    The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
                    where the error is in the fourth feature.
                    The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
                    where the errors are in the second and third features.  
    
            5. Number of Instances: 150 (50 in each of three classes)
    
            6. Number of Attributes: 4 numeric, predictive attributes and the class
    
            7. Attribute Information:
                1. sepal length in cm
                2. sepal width in cm
                3. petal length in cm
                4. petal width in cm
                5. class: 
                    -- Iris Setosa
                    -- Iris Versicolour
                    -- Iris Virginica
    
            8. Missing Attribute Values: None
    
            Summary Statistics:
                        Min  Max   Mean    SD   Class Correlation
               sepal length: 4.3  7.9   5.84  0.83    0.7826   
                sepal width: 2.0  4.4   3.05  0.43   -0.4194
               petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
                petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
    
            9. Class Distribution: 33.3% for each of 3 classes.
            



```python
print(iris.data.shape,iris.target.shape)
```

    (150, 4) (150, 1)



```python
print(iris.target_names)
```

    ['setosa', 'versicolor', 'virginica']



```python
print(iris.feature_names)
```

    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']



```python
print(iris.data[:5,:])
```

    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]



```python
print(iris.target[:5,:])
```

    [[0]
     [0]
     [0]
     [0]
     [0]]



```python

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
```


    
![png](static/output_174_0.png)
    



```python
dum_classifier = numpy.random.random((len(iris.target),3))
dum_classifier/=dum_classifier.sum(axis=1,keepdims=True)
print(dum_classifier.shape)
dum_classifier[:5,:]
```

    (150, 3)





    array([[0.30771342, 0.63291054, 0.05937604],
           [0.19948573, 0.3398386 , 0.46067567],
           [0.42137261, 0.06943003, 0.50919737],
           [0.08200258, 0.60102031, 0.31697711],
           [0.27441097, 0.40437459, 0.32121444]])




```python
from neural_net.pipeline import onehot
```


```python
y = onehot(iris.target)
print(y.shape)
y[:5,:]
```

    (150, 3)





    array([[1, 0, 0],
           [1, 0, 0],
           [1, 0, 0],
           [1, 0, 0],
           [1, 0, 0]])




```python
ce.compute(y,dum_classifier)
```




    1.2920221019539604




```python
round(ce.compute(y,y))
```




    0



#### Properties

##### binary case : 2 labels


```python
ps = numpy.linspace(1e-9,1-1e-9,1000).reshape(-1,1)

y1,pr = [],[]

for p in ps:
    y1.append(ce.compute(numpy.array([[1,0]]),numpy.c_[p,1-p]))
    pr.append(ce.pr()[0])

fig,ax1 = plt.subplots()
plt.title('y=[1,0]')
ax1.plot(ps,y1,label='[p0,p1=1-p0]')
ax1.legend()
```




    <matplotlib.legend.Legend at 0x7f00792e5310>




    
![png](static/output_182_1.png)
    



```python
pr=numpy.array(pr)
pr.shape
```




    (1000, 2)




```python
fig,ax1 = plt.subplots()
ax1.set_title('derivatives at y=[1,0]')
ax1.plot(ps[10:-10,:],pr[10:-10,0],label=r"$CE'_{p0}$")
ax1.plot(ps[10:-10,:],pr[10:-10,1],label=r"$CE'_{p1}$")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f0079144dd0>




    
![png](static/output_184_1.png)
    


##### Multimodal case : 3+ labels


```python
ps = numpy.linspace(1e-9,1-1e-9,1000).reshape(-1,1)


y1,pr = [],[]

for p in ps:
    y1.append(ce.compute(numpy.array([[0,1,0]]),numpy.c_[(1-p)*2/3,p,(1-p)/3]))
    pr.append(ce.pr()[0])

fig,ax1 = plt.subplots()
plt.title('y=[0,1,0]')
ax1.plot(ps,y1,label='[p0,p1,p2]')
ax1.legend()
```




    <matplotlib.legend.Legend at 0x7f0079016210>




    
![png](static/output_186_1.png)
    



```python
pr=numpy.array(pr)
pr.shape
```




    (1000, 3)




```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
ax1.set_title('derivatives at y=[0,1,0]')
ax1.plot(ps[10:-10,:],pr[10:-10,0],label=r"$CE'_{p0}$")
ax1.plot(ps[10:-10,:],pr[10:-10,1],label=r"$CE'_{p1}$")
ax1.plot(ps[10:-10,:],pr[10:-10,2],label=r"$CE'_{p2}$")

ax2.set_title('derivatives at y=[0,1,0]')
ax2.plot(ps[10:-10,:],pr[10:-10,0],label=r"$CE'_{p0}$")
ax2.plot(ps[10:-10,:],pr[10:-10,2],label=r"$CE'_{p2}$")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f0078ff72d0>




    
![png](static/output_188_1.png)
    


### Mean Squared Error


```python
mse = MSE()
```

#### Boston Housing


```python
housing = HouseDatasetDownloader()
housing.load_dataset()
```


```python
print(housing.description)
```

     The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
     prices and the demand for clean air', J. Environ. Economics & Management,
     vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
     ...', Wiley, 1980.   N.B. Various transformations are used in the table on
     pages 244-261 of the latter.
    
     Variables in order:
     CRIM     per capita crime rate by town
     ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
     INDUS    proportion of non-retail business acres per town
     CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
     NOX      nitric oxides concentration (parts per 10 million)
     RM       average number of rooms per dwelling
     AGE      proportion of owner-occupied units built prior to 1940
     DIS      weighted distances to five Boston employment centres
     RAD      index of accessibility to radial highways
     TAX      full-value property-tax rate per $10,000
     PTRATIO  pupil-teacher ratio by town
     B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
     LSTAT    % lower status of the population
     MEDV     Median value of owner-occupied homes in $1000's
    
     



```python
print(housing.columns)
```

    ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']



```python
print(housing.data)
```

    [[6.3200e-03 1.8000e+01 2.3100e+00 ... 3.9690e+02 4.9800e+00 2.4000e+01]
     [2.7310e-02 0.0000e+00 7.0700e+00 ... 3.9690e+02 9.1400e+00 2.1600e+01]
     [2.7290e-02 0.0000e+00 7.0700e+00 ... 3.9283e+02 4.0300e+00 3.4700e+01]
     ...
     [6.0760e-02 0.0000e+00 1.1930e+01 ... 3.9690e+02 5.6400e+00 2.3900e+01]
     [1.0959e-01 0.0000e+00 1.1930e+01 ... 3.9345e+02 6.4800e+00 2.2000e+01]
     [4.7410e-02 0.0000e+00 1.1930e+01 ... 3.9690e+02 7.8800e+00 1.1900e+01]]



```python
print(housing.data.shape)
```

    (506, 14)


##### Correlations


```python

fig = plt.figure(figsize=(22,14))

gs  = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

ax4 = fig.add_subplot(gs[1, :])



ax1.hist(housing.data[:,-1],label="Median value of owner-occupied homes in $1,000's",bins=10)
ax1.legend()
ax2.hist(((lp:=numpy.log(housing.data[:,-1]))-lp.mean())/lp.std(),label='Normalized Log Median House Values',bins=15)
ax2.legend()
corr = Pearson(housing.data,cols=housing.columns)
corr.corr()
ax4.scatter(x=housing.data[:,-2],y=housing.data[:,-1],label='MEDV by % lower status of the population' )

ax4.set_xlabel("LSTAT")
ax4.set_ylabel("MEDV")
ax4.legend()

corr.heatmap(ax=ax3,digits=2)


```


    
![png](static/output_198_0.png)
    


##### Ordinary Least Squares


```python
X = numpy.c_[housing.data[:,[-2]],numpy.ones((len(housing.data),1))]
y = housing.data[:,[-1]]
XtX = X.T.dot(X)
XTXinv = numpy.linalg.inv(XtX)  
βhat = XTXinv.dot(X.T.dot(y))

pred = X.dot(βhat)
ε = y-pred

sigmaε,meanε = ε.std(),ε.mean()
print(sigmaε,meanε)

Varβ = sigmaε**2 * XTXinv

βhat
```

    6.20346413142642 -1.842355868215912e-14





    array([[-0.95004935],
           [34.55384088]])



##### t-test


```python
student = βhat/Varβ.diagonal().reshape(-1,1)**.5
student
```




    array([[-24.57651813],
           [ 61.53688032]])




```python
import scipy.stats 
  
p_value = scipy.stats.norm.sf(abs(student)) 
p_value
```




    array([[1.12616355e-133],
           [0.00000000e+000]])



##### R2 score


```python
SSE = (ε**2).sum()
SStot = ((y-y.mean())**2).sum()
R2 = 1 - SSE/SStot
R2
```




    0.5441462975864797




```python
plt.scatter(x=housing.data[:,-2],y=housing.data[:,-1],label='MEDV by % lower status of the population' )
plt.plot(housing.data[:,-2],pred,label="y=$34,000 - \$950xLSTAT;\ R^{2}=$"+f'{R2:.2}')
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f4d64160450>




    
![png](static/output_206_1.png)
    


##### Computing mse


```python
mse = MSE()
```


```python
mse.compute(y,pred)
```




    38.48296722989415




```python
mse.compute(y,y)
```




    0.0



##### Residual Analysis


```python
probs = numpy.linspace(0,1,100)
εquantiles = numpy.quantile(ε,probs)
theoratical = numpy.random.normal(loc=meanε,scale=sigmaε,size=10000)
normal_quantiles = numpy.quantile(theoratical,probs)
```


```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(22,7))

ax1.scatter(normal_quantiles,εquantiles,label='Residual quantiles')
ax1.plot(normal_quantiles,normal_quantiles,label='theoratical normal')
ax1.set_title("Normal Q-Q")
ax1.set_xlabel("normal quantiles")
ax1.set_ylabel("ε quantiles")


ax2.scatter(pred,ε,label=r"Residuals")
ax2.set_title(r"Residuals vs fitted")
ax2.set_xlabel("predictions")
ax2.set_ylabel("ε")
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d54953290>




    
![png](static/output_213_1.png)
    


##### Biais analysis


```python
coeffs = []
for (x_new,y_new) in Boostrap((X,y),n_sample=1000):
    β_new = numpy.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T.dot(y_new))
    coeffs.append(β_new)
```


```python
coeffs = numpy.concatenate(coeffs,axis=1).T
```


```python
coeffs.mean(axis=0)
```




    array([-0.95607794, 34.59180802])




```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(22,7))


ax1.hist(coeffs[:,0],label=r"$\beta_{LSTAT}$",bins=15)
ax1.set_title(r"$E(\beta_{LSTAT})=-\$957$")
ax1.legend()
ax2.hist(coeffs[:,1],label=r"$\beta_{Intercept}$",bins=15)
ax2.set_title(r"$E(\beta_{Intercept})=\$34,625$")
ax2.legend()

```




    <matplotlib.legend.Legend at 0x7f4d545e6610>




    
![png](static/output_218_1.png)
    


#### Properties


```python
ps = numpy.linspace(0,200,1000).reshape(-1,1)

mses = numpy.array([ [mse.compute(numpy.array([100]),p),mse.pr()[0]] for p in ps ])

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
ax1.plot(ps,mses[:,0],label='mse')

ax1.set_title('mse for true=100')
ax1.legend()

ax2.plot(ps,mses[:,1],label=r'$-2(y_{n,1}-p_{n,1})$')
ax2.set_title('mse derivative at true=100')
ax2.legend()
```




    <matplotlib.legend.Legend at 0x7f4d54150950>




    
![png](static/output_220_1.png)
    


## Backpropagation


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ
from neural_net.cost import BinaryCrossEntropy
from neural_net.utils import make_circle_data
import numpy
import matplotlib.pyplot as plt
```


```python
network = Sequential(
        [
        Fullyconnected(2,10,zeros),
        Activation(σ),
        Fullyconnected(10,1,zeros),
        Activation(σ)
        ]
    ,BinaryCrossEntropy,store=True)
```


```python
network['steps']
```




    [Fullyconnected, Activation, Fullyconnected, Activation]




```python
n,k = 1000,2
X = numpy.random.uniform(-100,100,size=(n,k))
y = (X.sum(axis=1) < numpy.random.uniform(30,90,(len(X),))).reshape(-1,1)+0
plt.scatter(x=X[:,0],y=X[:,1],c=y)
```




    <matplotlib.collections.PathCollection at 0x7f792bffff10>




    
![png](static/output_225_1.png)
    



```python
network.predict(X)
network.out.shape
```




    (1000, 1)




```python
network['cost']
```




    BinaryCrossEntropy




```python
network['cost'].compute(y,network.out)
```




    0.6931471805599454




```python
Δ0 = network['cost'].pr()
Δ0.shape
```




    (1000, 1)



### Output Layer


```python
network['steps'][-1]
```




    Activation




```python
Δ1 = network['steps'][-1].func.grad(Δ0)
Δ1.shape
```




    (1000, 1)



### Last Linear Layer


```python
network['steps'][-2]
```




    Fullyconnected




```python
Δ2 = network['steps'][-2].func.grad(Δ1)
Δ2.shape
```




    (1000, 10)




```python
%load_ext sql
%sql $network.db_path
```

### Update method


```python
network.update(Δ0)
```




    array([[ 0.34141679,  0.29918978],
           [ 0.34141679,  0.29918978],
           [-0.34141679, -0.29918978],
           ...,
           [ 0.34141679,  0.29918978],
           [ 0.34141679,  0.29918978],
           [ 0.34141679,  0.29918978]])



### View weights changes on db


```python
network.updateW()
network.commit()
```


```sql
%%sql
SELECT *
from weight
```

     * sqlite:////home/analyst/notebooks/module/neural_net/run/model1709931963.db
    Done.





<table>
    <thead>
        <tr>
            <th>value</th>
            <th>Neurons_id</th>
            <th>id</th>
            <th>created_at</th>
            <th>updated_at</th>
            <th>name</th>
            <th>Weight_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>1</td>
            <td>2024-03-08 21:06:03.650976</td>
            <td>2024-03-08 21:07:12.085321</td>
            <td>None</td>
            <td>0_0</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>2</td>
            <td>2024-03-08 21:06:03.650981</td>
            <td>2024-03-08 21:07:12.085329</td>
            <td>None</td>
            <td>0_1</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>3</td>
            <td>2024-03-08 21:06:03.650983</td>
            <td>2024-03-08 21:07:12.085330</td>
            <td>None</td>
            <td>0_2</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>4</td>
            <td>2024-03-08 21:06:03.650984</td>
            <td>2024-03-08 21:07:12.085332</td>
            <td>None</td>
            <td>0_3</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>5</td>
            <td>2024-03-08 21:06:03.650985</td>
            <td>2024-03-08 21:07:12.085332</td>
            <td>None</td>
            <td>0_4</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>6</td>
            <td>2024-03-08 21:06:03.650985</td>
            <td>2024-03-08 21:07:12.085333</td>
            <td>None</td>
            <td>0_5</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>7</td>
            <td>2024-03-08 21:06:03.650987</td>
            <td>2024-03-08 21:07:12.085333</td>
            <td>None</td>
            <td>0_6</td>
        </tr>
        <tr>
            <td>-0.9932124687481962</td>
            <td>140158405172944</td>
            <td>8</td>
            <td>2024-03-08 21:06:03.650987</td>
            <td>2024-03-08 21:07:12.085334</td>
            <td>None</td>
            <td>0_7</td>
        </tr>
        <tr>
            <td>-0.9932124687481961</td>
            <td>140158405172944</td>
            <td>9</td>
            <td>2024-03-08 21:06:03.650988</td>
            <td>2024-03-08 21:07:12.085334</td>
            <td>None</td>
            <td>0_8</td>
        </tr>
        <tr>
            <td>-0.9932124687481961</td>
            <td>140158405172944</td>
            <td>10</td>
            <td>2024-03-08 21:06:03.650989</td>
            <td>2024-03-08 21:07:12.085335</td>
            <td>None</td>
            <td>0_9</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>11</td>
            <td>2024-03-08 21:06:03.650990</td>
            <td>2024-03-08 21:07:12.085336</td>
            <td>None</td>
            <td>1_0</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>12</td>
            <td>2024-03-08 21:06:03.650991</td>
            <td>2024-03-08 21:07:12.085336</td>
            <td>None</td>
            <td>1_1</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>13</td>
            <td>2024-03-08 21:06:03.650992</td>
            <td>2024-03-08 21:07:12.085337</td>
            <td>None</td>
            <td>1_2</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>14</td>
            <td>2024-03-08 21:06:03.650993</td>
            <td>2024-03-08 21:07:12.085337</td>
            <td>None</td>
            <td>1_3</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>15</td>
            <td>2024-03-08 21:06:03.650994</td>
            <td>2024-03-08 21:07:12.085338</td>
            <td>None</td>
            <td>1_4</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>16</td>
            <td>2024-03-08 21:06:03.650995</td>
            <td>2024-03-08 21:07:12.085339</td>
            <td>None</td>
            <td>1_5</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>17</td>
            <td>2024-03-08 21:06:03.650996</td>
            <td>2024-03-08 21:07:12.085339</td>
            <td>None</td>
            <td>1_6</td>
        </tr>
        <tr>
            <td>-0.8703702558942491</td>
            <td>140158405172944</td>
            <td>18</td>
            <td>2024-03-08 21:06:03.650997</td>
            <td>2024-03-08 21:07:12.085340</td>
            <td>None</td>
            <td>1_7</td>
        </tr>
        <tr>
            <td>-0.8703702558942492</td>
            <td>140158405172944</td>
            <td>19</td>
            <td>2024-03-08 21:06:03.650998</td>
            <td>2024-03-08 21:07:12.085340</td>
            <td>None</td>
            <td>1_8</td>
        </tr>
        <tr>
            <td>-0.8703702558942492</td>
            <td>140158405172944</td>
            <td>20</td>
            <td>2024-03-08 21:06:03.650999</td>
            <td>2024-03-08 21:07:12.085341</td>
            <td>None</td>
            <td>1_9</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>21</td>
            <td>2024-03-08 21:06:03.651000</td>
            <td>2024-03-08 21:07:12.085341</td>
            <td>None</td>
            <td>2_0</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>22</td>
            <td>2024-03-08 21:06:03.651001</td>
            <td>2024-03-08 21:07:12.085342</td>
            <td>None</td>
            <td>2_1</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>23</td>
            <td>2024-03-08 21:06:03.651002</td>
            <td>2024-03-08 21:07:12.085343</td>
            <td>None</td>
            <td>2_2</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>24</td>
            <td>2024-03-08 21:06:03.651003</td>
            <td>2024-03-08 21:07:12.085343</td>
            <td>None</td>
            <td>2_3</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>25</td>
            <td>2024-03-08 21:06:03.651004</td>
            <td>2024-03-08 21:07:12.085344</td>
            <td>None</td>
            <td>2_4</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>26</td>
            <td>2024-03-08 21:06:03.651005</td>
            <td>2024-03-08 21:07:12.085344</td>
            <td>None</td>
            <td>2_5</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>27</td>
            <td>2024-03-08 21:06:03.651006</td>
            <td>2024-03-08 21:07:12.085345</td>
            <td>None</td>
            <td>2_6</td>
        </tr>
        <tr>
            <td>0.01890625000000001</td>
            <td>140158405172944</td>
            <td>28</td>
            <td>2024-03-08 21:06:03.651006</td>
            <td>2024-03-08 21:07:12.085346</td>
            <td>None</td>
            <td>2_7</td>
        </tr>
        <tr>
            <td>0.01890625</td>
            <td>140158405172944</td>
            <td>29</td>
            <td>2024-03-08 21:06:03.651007</td>
            <td>2024-03-08 21:07:12.085346</td>
            <td>None</td>
            <td>2_8</td>
        </tr>
        <tr>
            <td>0.01890625</td>
            <td>140158405172944</td>
            <td>30</td>
            <td>2024-03-08 21:06:03.651008</td>
            <td>2024-03-08 21:07:12.085347</td>
            <td>None</td>
            <td>2_9</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>31</td>
            <td>2024-03-08 21:06:03.651009</td>
            <td>2024-03-08 21:07:12.085347</td>
            <td>None</td>
            <td>0_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>32</td>
            <td>2024-03-08 21:06:03.651010</td>
            <td>2024-03-08 21:07:12.085348</td>
            <td>None</td>
            <td>1_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>33</td>
            <td>2024-03-08 21:06:03.651011</td>
            <td>2024-03-08 21:07:12.085348</td>
            <td>None</td>
            <td>2_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>34</td>
            <td>2024-03-08 21:06:03.651012</td>
            <td>2024-03-08 21:07:12.085349</td>
            <td>None</td>
            <td>3_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>35</td>
            <td>2024-03-08 21:06:03.651013</td>
            <td>2024-03-08 21:07:12.085349</td>
            <td>None</td>
            <td>4_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>36</td>
            <td>2024-03-08 21:06:03.651014</td>
            <td>2024-03-08 21:07:12.085350</td>
            <td>None</td>
            <td>5_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>37</td>
            <td>2024-03-08 21:06:03.651015</td>
            <td>2024-03-08 21:07:12.085351</td>
            <td>None</td>
            <td>6_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>38</td>
            <td>2024-03-08 21:06:03.651016</td>
            <td>2024-03-08 21:07:12.085351</td>
            <td>None</td>
            <td>7_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>39</td>
            <td>2024-03-08 21:06:03.651017</td>
            <td>2024-03-08 21:07:12.085352</td>
            <td>None</td>
            <td>8_0</td>
        </tr>
        <tr>
            <td>0.275</td>
            <td>140158405279056</td>
            <td>40</td>
            <td>2024-03-08 21:06:03.651018</td>
            <td>2024-03-08 21:07:12.085354</td>
            <td>None</td>
            <td>9_0</td>
        </tr>
        <tr>
            <td>0.55</td>
            <td>140158405279056</td>
            <td>41</td>
            <td>2024-03-08 21:06:03.651019</td>
            <td>2024-03-08 21:07:12.085354</td>
            <td>None</td>
            <td>10_0</td>
        </tr>
    </tbody>
</table>



## Logistic Regression
![png](static/86d36e2f-537d-43ec-9d71-4d37543513a2.png)


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ,Softmax
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.utils import make_circle_data
import numpy
import matplotlib.pyplot as plt

```


```python
centers = [(-50, 0), (20, 30)]
radii = [30, 35]
X, y = make_circle_data(centers, radii)
print(X.shape, y.shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=y)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
```

    (328, 2) (328, 1)





    (-100.0, 100.0)




    
![png](static/output_244_2.png)
    


### Pure numpy definition


```python
H_θ = lambda X,θ : 1/(1+numpy.exp(-X.dot(θ)))
θ = numpy.zeros((3,1))
X_const = numpy.c_[X,numpy.ones((len(X),1))]
H_θ(X_const,θ).shape
```




    (328, 1)



### Using layers


```python
LogReg = Sequential(
        [
        Fullyconnected(2,1,zeros),
        Activation(σ)
        ],
    BinaryCrossEntropy
    )
LogReg.predict(X).shape
```




    (328, 1)




```python
LogReg['steps'][-2].func.W
```




    array([[0.],
           [0.],
           [0.]])



### Computing Gradients

#### Analytic gradient
![png](static/751eae42-4f94-4fff-a583-3f05b9eb45b0.png)


```python
J = lambda θ,X : 1/len(X)*X.T.dot(H_θ(X,θ)-y)
J(θ,X_const)
```




    array([[ -8.2471052 ],
           [-15.8967726 ],
           [ -0.07317073]])



#### Chain rule logistic Regression



```python
LogReg["cost"].compute(y,LogReg.out)
p0 =  LogReg["cost"].pr()
p1 = LogReg["steps"][-1].func.grad(p0)
p2 = LogReg["steps"][-2].func.grad(p1)
```


```python
-LogReg["steps"][-2].func.W
```




    array([[ -8.2471052 ],
           [-15.8967726 ],
           [ -0.07317073]])



## Softmax Regression( j target class in {1,..,k} and m=1,..,n instances)

$$
{\displaystyle \sigma (\mathbf {z} )_{j}={\frac {\mathrm {e} ^{z_{j}}}{\sum _{i=1}^{K}\mathrm {e} ^{z_{i}}}}}
$$

### Analytical gradient


```python
from neural_net.pipeline import onehot
```


```python
y_one_hot = onehot(y)
y_one_hot[:5,:]
```




    array([[1, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [1, 0]])




```python
W = numpy.zeros((X.shape[1]+1,2))
W
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])




```python
W = numpy.zeros((X.shape[1],2))
Sm = lambda W,X : numpy.exp(X.dot(W))/numpy.exp(X.dot(W)).sum(axis=1).reshape(-1,1)
Sm(W,X)[:5,:]
```




    array([[0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5]])




```python
J = 1/len(X)*X.T.dot(Sm(W,X)-y_one_hot)
J
```




    array([[  8.2471052,  -8.2471052],
           [ 15.8967726, -15.8967726]])



#### Chain rule


```python
softmax = Sequential(
        [
        Fullyconnected(2,2,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )
softmax.predict(X)[:5,:]
```




    array([[0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5],
           [0.5, 0.5]])




```python
softmax["cost"].compute(y_one_hot,softmax.out)
```




    0.6931471805599453




```python
softmax["cost"].compute(y_one_hot,softmax.out)
_ = softmax.update(softmax["cost"].pr())
-softmax["steps"][-2].func.W
```




    array([[ -8.2471052 ,   8.2471052 ],
           [-15.8967726 ,  15.8967726 ],
           [ -0.07317073,   0.07317073]])



## Training


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ,Softmax
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.utils import make_circle_data,IrisDatasetDownloader
from neural_net.metrics import accuracy
from neural_net.pipeline import onehot
import numpy
import matplotlib.pyplot as plt
```


```python
centers = [(-50, 0), (20, 30)]
radii = [30, 35]
X, y = make_circle_data(centers, radii)
print(X.shape, y.shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=y)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
```

    (328, 2) (328, 1)





    (-100.0, 100.0)




    
![png](static/output_271_2.png)
    


### Logistic Regression


```python
n_epoch = 1000
α = 0.1
```

#### Using analytical gradient


```python
H_θ = lambda X,θ : 1/(1+numpy.exp(-X.dot(θ)))
θ = numpy.zeros((3,1))
X_const = numpy.c_[X,numpy.ones((len(X),1))]
J = lambda θ,X : 1/len(X)*X.T.dot(H_θ(X,θ)-y)
```


```python
for _ in range(n_epoch):
    θ -= α*J(θ,X_const)
```


```python
θ
```




    array([[1.19524694],
           [1.47756754],
           [0.05197856]])




```python
pred = (H_θ(X_const,θ) > .5 )+0

```


```python
plt.scatter(x=X[:,0],y=X[:,1],c=pred)
```




    <matplotlib.collections.PathCollection at 0x7f2c4820df90>




    
![png](static/output_279_1.png)
    


#### Using chain rule


```python
LogReg = Sequential(
        [
        Fullyconnected(2,1,zeros),
        Activation(σ)
        ],
    BinaryCrossEntropy
    )
for _ in range(n_epoch):
    LogReg.predict(X)
    LogReg["cost"].compute(y,LogReg.out)
    LogReg["cost"].compute(y,LogReg.out)
    _ = LogReg.update(α*LogReg["cost"].pr())
```


```python
LogReg['steps'][-2].func.W
```




    array([[1.19524694],
           [1.47756754],
           [0.05197856]])




```python
p = (LogReg.out>.5)+0
```


```python
plt.scatter(x=X[:,0],y=X[:,1],c=p)
```




    <matplotlib.collections.PathCollection at 0x7f2c481c2b90>




    
![png](static/output_284_1.png)
    


#### Using train method


```python
LogReg = Sequential(
        [
        Fullyconnected(2,1,zeros),
        Activation(σ)
        ],
    BinaryCrossEntropy
    )
LogReg.train(X,y,epochs=n_epoch,α=α,metrics=accuracy)
```

    BinaryCrossEntropy 0.0002 accuracy 1.0: 100%|██████████| 1000/1000 [00:03<00:00, 298.57it/s]



```python
LogReg['steps'][-2].func.W
```




    array([[1.19524694],
           [1.47756754],
           [0.05197856]])



### Softmax

#### 2 labels


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ,Softmax
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.utils import make_circle_data,IrisDatasetDownloader
from neural_net.metrics import accuracy
from neural_net.pipeline import onehot
import numpy
import matplotlib.pyplot as plt
```


```python
centers = [(-50, 0), (20, 30)]
radii = [30, 35]
X, y = make_circle_data(centers, radii)
y = onehot(y)
print(X.shape, y.shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=y.argmax(axis=1))
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
```

    (328, 2) (328, 2)





    (-100.0, 100.0)




    
![png](static/output_291_2.png)
    



```python
n_epoch = 1000
α = 0.1
```

##### Analytically


```python
n,k=X.shape
j=y.shape[1]
```


```python
W = numpy.zeros((k+1,j))
Sm = lambda W,X : numpy.exp(X.dot(W))/numpy.exp(X.dot(W)).sum(axis=1).reshape(-1,1)
J_W = lambda W,X : 1/n*X.T.dot(Sm(W,X)-y)

for _ in range(n_epoch):
    W -= α*J_W(W,numpy.c_[X,numpy.ones((n,1))] )
```


```python
W
```




    array([[-1.07789341,  1.07789341],
           [-1.56804297,  1.56804297],
           [-0.02426586,  0.02426586]])




```python
p = Sm(W,numpy.c_[X,numpy.ones((n,1))]).argmax(axis=1)
plt.scatter(x=X[:,0],y=X[:,1],c=p)
```




    <matplotlib.collections.PathCollection at 0x7f0083e3af50>




    
![png](static/output_297_1.png)
    


##### using chain rule


```python
softmax = Sequential(
        [
        Fullyconnected(k,j,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )
softmax.train(X,y,epochs=n_epoch,α=α,metrics=accuracy)
```

    CrossEntropy 0.0001 accuracy 1.0: 100%|██████████| 1000/1000 [00:03<00:00, 282.21it/s]



```python
softmax['steps'][-2].func.W
```




    array([[-1.07789341,  1.07789341],
           [-1.56804297,  1.56804297],
           [-0.02426586,  0.02426586]])




```python
p = softmax.predict(X).argmax(axis=1)
plt.scatter(x=X[:,0],y=X[:,1],c=p)
```




    <matplotlib.collections.PathCollection at 0x7f0083de2e90>




    
![png](static/output_301_1.png)
    


#### 3 labels


```python
iris = IrisDatasetDownloader()
iris.load_dataset()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
```


    
![png](static/output_303_0.png)
    



```python
X,y = iris.data,onehot(iris.target)
n,k = X.shape
j = y.shape[1]
X.shape,y.shape
```




    ((150, 4), (150, 3))



##### Using Softmax analytical solution


```python
n_epoch=1000
```


```python
W = numpy.zeros((k+1,j))
Sm = lambda W,X : numpy.exp(X.dot(W))/numpy.exp(X.dot(W)).sum(axis=1).reshape(-1,1)
J_W = lambda W,X : 1/n*X.T.dot(Sm(W,X)-y)

for _ in range(n_epoch):
    W -= α*J_W(W,numpy.c_[X,numpy.ones((n,1))]) 
```


```python
W
```




    array([[ 0.88907029,  0.7230017 , -1.612072  ],
           [ 2.05504822, -0.20042131, -1.85462692],
           [-2.81867225, -0.13349788,  2.95217013],
           [-1.32033636, -1.14715458,  2.46749093],
           [ 0.42390714,  0.65830856, -1.0822157 ]])




```python
p = Sm(W,numpy.c_[X,numpy.ones((n,1))]).argmax(axis=1)
plt.scatter(x=X[:,0],y=X[:,1],c=p) 
```




    <matplotlib.collections.PathCollection at 0x7f008403af50>




    
![png](static/output_309_1.png)
    



```python
accuracy().compute(y,Sm(W,numpy.c_[X,numpy.ones((n,1))]))
```




    0.9867



##### Chain rule


```python
softmax = Sequential(
        [
        Fullyconnected(k,j,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )
softmax.train(X,y,epochs=n_epoch,α=α,metrics=accuracy)
```

    CrossEntropy 0.126 accuracy 0.9867: 100%|██████████| 1000/1000 [00:03<00:00, 318.95it/s]



```python
p = softmax.predict(X).argmax(axis=1)
plt.scatter(x=X[:,0],y=X[:,1],c=p)
```




    <matplotlib.collections.PathCollection at 0x7f0080252e90>




    
![png](static/output_313_1.png)
    


## Non Linear Problems


```python
import numpy
```


```python
n,k = 1500,2
X = numpy.random.uniform(-100,100,size=(n,k))
y =( (X[:, 0]**2 + X[:, 1]**2) < 3000).reshape(-1,1)+0
y_one_hot = onehot(y)
plt.scatter(x=X[:,0],y=X[:,1],c=y)
```




    <matplotlib.collections.PathCollection at 0x7f0d3d72b510>




    
![png](static/output_316_1.png)
    


## Beyond linear architecture


```python
n_epoch = 1000
α = 0.2
```

### Logistic regression


```python
H_θ = lambda X,θ : 1/(1+numpy.exp(-X.dot(θ)))
θ = numpy.zeros((3,1))
X_const = numpy.c_[X,numpy.ones((len(X),1))]
J = lambda θ,X : 1/len(X)*X.T.dot(H_θ(X,θ)-y)
for _ in range(n_epoch):
    θ -= α*J(θ,X_const)
```


```python
pred = (H_θ(X_const,θ) > .5 )+0

```


```python
(pred==y).sum()/len(y)
```




    0.49066666666666664




```python
plt.scatter(x=X[:,0],y=X[:,1],c=pred)
```




    <matplotlib.collections.PathCollection at 0x7f00797c7e10>




    
![png](static/output_323_1.png)
    


### Softmax


```python
y_one_hot = onehot(y)
n,j= y_one_hot.shape
(n,j)
```




    (1500, 2)




```python
W = numpy.zeros((k+1,j))
Sm = lambda W,X : numpy.exp(X.dot(W))/numpy.exp(X.dot(W)).sum(axis=1).reshape(-1,1)
J_W = lambda W,X : 1/n*X.T.dot(Sm(W,X)-y_one_hot)

for _ in range(n_epoch):
    W -= α*J_W(W,X_const) 
```


```python
p = Sm(W,X_const).argmax(axis=1)
plt.scatter(x=X[:,0],y=X[:,1],c=p) 
```




    <matplotlib.collections.PathCollection at 0x7f007949a8d0>




    
![png](static/output_327_1.png)
    


## Neural network


```python
from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros,XavierHe
from neural_net.activation import σ,Softmax,LeakyReLU,Tanh,ELU,ReLU
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.metrics import accuracy
from neural_net.pipeline import onehot,scaler,shuffle,Batch
from neural_net.utils import IrisDatasetDownloader
import numpy
import matplotlib.pyplot as plt
```

### Xavier and He Initialization methods

We don’t want the signal to die out, nor do we want it to explode and saturate.

For the signal to flow properly, the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs



```python
NN = Sequential(
        [
        Fullyconnected(2,50,XavierHe("Uniform","ReLU").init_func),
        Activation(LeakyReLU),     
        Fullyconnected(50,1,XavierHe("Uniform","Sigmoid").init_func),
        Activation(σ)
        ],
    BinaryCrossEntropy
    )
```


```python
NN.train(scaler(X),y,α=α,epochs=n_epoch,metrics=accuracy)
```

    BinaryCrossEntropy 0.0721 accuracy 0.994: 100%|██████████| 1000/1000 [00:17<00:00, 56.29it/s]



```python
pred = (NN.predict(scaler(X))>.5)+0
pred
```




    array([[0],
           [1],
           [0],
           ...,
           [0],
           [0],
           [0]])




```python
plt.scatter(x=X[:,0],y=X[:,1],c=pred)
```




    <matplotlib.collections.PathCollection at 0x7f0d3b510590>




    
![png](static/output_334_1.png)
    



```python
NN.train(scaler(X),y,α=α,epochs=n_epoch,metrics=accuracy)
```

    BinaryCrossEntropy 0.0463 accuracy 0.996: 100%|██████████| 1000/1000 [00:19<00:00, 51.83it/s]


### Iris Problem


```python
iris = IrisDatasetDownloader()
iris.load_dataset()
y = onehot(iris.target)
X = iris.data

X,y = shuffle(X,y)
n,k = X.shape
j = y.shape[1]
```


```python
NN = Sequential(
        [
        Fullyconnected(k,100,XavierHe("Uniform","ReLU").init_func),
        Activation(LeakyReLU),  
        Fullyconnected(100,50,XavierHe("Normal","ReLU").init_func),
        Activation(ELU),  
        Fullyconnected(50,j,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )
```


```python
batch = Batch(60,len(X),lambda : scaler(X), lambda : y)
NN.train(batch=batch,α=0.2,epochs=1000,metrics=accuracy)
```

    CrossEntropy 0.001 accuracy 1.0: 100%|██████████| 1000/1000 [00:32<00:00, 30.43it/s]  



```python
pred = NN.predict(scaler(X)).argmax(axis=1,keepdims=True)
```


```python
true_y = y.argmax(axis=1,keepdims=True)
```


```python
(pred == true_y).sum()/len(pred)
```




    1.0



### Storing weights


```python
from neural_net.model import Define
Define._Define__store = True
```


```python
NN.updateW()
```


```python
NN.commit()
```


```python
NN.db_path
```




    'sqlite:////home/analyst/notebooks/module/neural_net/run/model1709981118.db'




```python
from sqlalchemy import text
import pandas
cursor = NN.engines.get(NN.db_path).connect()
res = cursor.execute(text('''

        SELECT * 
        FROM
        Weight

'''))
pandas.DataFrame(res.fetchall())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
      <th>Neurons_id</th>
      <th>id</th>
      <th>created_at</th>
      <th>updated_at</th>
      <th>name</th>
      <th>Weight_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.003729</td>
      <td>139694848570384</td>
      <td>1</td>
      <td>2024-03-09 11:08:51.902537</td>
      <td>2024-03-09 11:08:51.902543</td>
      <td>None</td>
      <td>0_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.384036</td>
      <td>139694848570384</td>
      <td>2</td>
      <td>2024-03-09 11:08:51.902544</td>
      <td>2024-03-09 11:08:51.902545</td>
      <td>None</td>
      <td>0_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.033767</td>
      <td>139694848570384</td>
      <td>3</td>
      <td>2024-03-09 11:08:51.902545</td>
      <td>2024-03-09 11:08:51.902546</td>
      <td>None</td>
      <td>0_2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.265035</td>
      <td>139694848570384</td>
      <td>4</td>
      <td>2024-03-09 11:08:51.902546</td>
      <td>2024-03-09 11:08:51.902547</td>
      <td>None</td>
      <td>0_3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.261001</td>
      <td>139694848570384</td>
      <td>5</td>
      <td>2024-03-09 11:08:51.902547</td>
      <td>2024-03-09 11:08:51.902548</td>
      <td>None</td>
      <td>0_4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>-0.207245</td>
      <td>139694847460432</td>
      <td>197</td>
      <td>2024-03-09 11:08:51.902896</td>
      <td>2024-03-09 11:08:51.902896</td>
      <td>None</td>
      <td>46_0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-0.108507</td>
      <td>139694847460432</td>
      <td>198</td>
      <td>2024-03-09 11:08:51.902896</td>
      <td>2024-03-09 11:08:51.902897</td>
      <td>None</td>
      <td>47_0</td>
    </tr>
    <tr>
      <th>198</th>
      <td>-0.229139</td>
      <td>139694847460432</td>
      <td>199</td>
      <td>2024-03-09 11:08:51.902897</td>
      <td>2024-03-09 11:08:51.902898</td>
      <td>None</td>
      <td>48_0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0.260771</td>
      <td>139694847460432</td>
      <td>200</td>
      <td>2024-03-09 11:08:51.902899</td>
      <td>2024-03-09 11:08:51.902899</td>
      <td>None</td>
      <td>49_0</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.265368</td>
      <td>139694847460432</td>
      <td>201</td>
      <td>2024-03-09 11:08:51.902900</td>
      <td>2024-03-09 11:08:51.902900</td>
      <td>None</td>
      <td>50_0</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 7 columns</p>
</div>



## Optical character recognition(OCR)

### Hand written dataset


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
```


```python
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(13, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title("Training: %i" % label)
```


    
![png](static/output_352_0.png)
    



```python
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

## Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, onehot(digits.target.reshape(-1,1)), test_size=0.5, shuffle=False
)
```


```python
X_train.shape,y_train.shape
```




    ((898, 64), (898, 10))



#### Softmax


```python
softmax = Sequential(
        [
        Fullyconnected(64,10,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )
softmax.train(X_train,y_train,epochs=1000,α=0.001,metrics=accuracy)
```

    CrossEntropy 0.1366 accuracy 0.9811: 100%|==========| 1000/1000 [00:25<00:00, 39.08it/s]



```python
predicted = softmax.predict(X_test).argmax(axis=1)
```


```python
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
```


    
![png](static/output_358_0.png)
    



```python
print(
    f"Classification report for classifier :\n"
    f"{classification_report(y_test.argmax(axis=1), predicted)}\n"
)
```

    Classification report for classifier :
                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98        88
               1       0.93      0.84      0.88        91
               2       1.00      0.97      0.98        86
               3       0.94      0.85      0.89        91
               4       0.97      0.91      0.94        92
               5       0.89      0.93      0.91        91
               6       0.93      0.99      0.96        91
               7       0.96      0.99      0.97        89
               8       0.89      0.90      0.89        88
               9       0.83      0.96      0.89        92
    
        accuracy                           0.93       899
       macro avg       0.93      0.93      0.93       899
    weighted avg       0.93      0.93      0.93       899
    
    



```python
disp = ConfusionMatrixDisplay.from_predictions(y_test.argmax(axis=1), predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
```

    Confusion matrix:
    [[85  0  0  0  1  1  1  0  0  0]
     [ 0 76  0  1  2  1  1  0  0 10]
     [ 0  0 83  3  0  0  0  0  0  0]
     [ 0  2  0 77  0  3  0  4  5  0]
     [ 0  0  0  0 84  0  4  0  3  1]
     [ 0  0  0  0  0 85  1  0  0  5]
     [ 0  1  0  0  0  0 90  0  0  0]
     [ 0  0  0  0  0  0  0 88  1  0]
     [ 0  3  0  0  0  4  0  0 79  2]
     [ 0  0  0  1  0  2  0  0  1 88]]



    
![png](static/output_360_1.png)
    


### Deep Learning


```python
NN = Sequential(
        [
        Fullyconnected(64,1000,XavierHe("Normal","ReLU").init_func),
        Activation(ELU),  
        Fullyconnected(1000,100,XavierHe("Normal","ReLU").init_func),
        Activation(ELU),  
        Fullyconnected(100,10,XavierHe("Normal","Sigmoid").init_func),
        Activation(Softmax)
        ],
    CrossEntropy
    )
```


```python
X_train,y_train = shuffle(X_train,y_train)
batch = Batch(10,len(X_train),lambda : X_train/16, lambda : y_train)
NN.train(batch=batch,α=0.014,epochs=100,metrics=accuracy)
```

    CrossEntropy 0.0117 accuracy 1.0: 100%|██████████| 100/100 [00:42<00:00,  2.35it/s]



```python
predicted = NN.predict(X_test/16).argmax(axis=1)
```


```python
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
```


    
![png](static/output_365_0.png)
    



```python
print(
    f"Classification report for classifier :\n"
    f"{classification_report(y_test.argmax(axis=1), predicted)}\n"
)
```

    Classification report for classifier :
                  precision    recall  f1-score   support
    
               0       1.00      0.98      0.99        88
               1       0.95      0.91      0.93        91
               2       0.99      1.00      0.99        86
               3       0.97      0.86      0.91        91
               4       0.98      0.92      0.95        92
               5       0.91      0.95      0.92        91
               6       0.95      0.99      0.97        91
               7       0.96      0.96      0.96        89
               8       0.92      0.92      0.92        88
               9       0.85      0.97      0.90        92
    
        accuracy                           0.94       899
       macro avg       0.95      0.94      0.94       899
    weighted avg       0.95      0.94      0.94       899
    
    


### Mnist


```python
from keras.datasets import mnist
```

    2024-03-09 13:59:33.278357: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-09 13:59:33.373835: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```


```python
num = 10
images = X_train[:num]
labels = Y_train[:num]
num_row = 2
num_col = 5

## plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()
```


    
![png](static/output_370_0.png)
    



```python
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
X_train /= 255
X_test /= 255
```


```python
X_train.shape
```




    (60000, 784)




```python
Y_train = onehot(Y_train.reshape(-1,1))
Y_train.shape, Y_test.shape
```




    ((60000, 10), (10000,))




```python
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
```


```python
NN = Sequential(
        [
        Fullyconnected(n_inputs,n_hidden1,XavierHe("Normal","ReLU").init_func),
        Activation(LeakyReLU),  
        Fullyconnected(n_hidden1,n_hidden2,XavierHe("Normal","ReLU").init_func),
        Activation(LeakyReLU),  
        Fullyconnected(n_hidden2,n_outputs,XavierHe("Normal","Sigmoid").init_func),
        Activation(Softmax)
        ],
    CrossEntropy
    )
```


```python
batch = Batch(500,len(X_train),lambda : X_train, lambda : Y_train)
NN.train(batch=batch,α=0.014,epochs=100,metrics=accuracy)
```

    CrossEntropy 0.129 accuracy 0.976: 100%|==========| 100/100 [10:28<00:00,  6.28s/it]



```python
pred = NN.predict(X_test).argmax(axis=1)
```


```python
pred
```




    array([7, 2, 1, ..., 4, 5, 6])




```python
print(
    f"Classification report for classifier :\n"
    f"{classification_report(Y_test, pred)}\n"
)
```

    Classification report for classifier :
                  precision    recall  f1-score   support
    
               0       0.98      0.99      0.98       980
               1       0.98      0.99      0.98      1135
               2       0.97      0.97      0.97      1032
               3       0.96      0.97      0.96      1010
               4       0.97      0.97      0.97       982
               5       0.95      0.96      0.96       892
               6       0.97      0.97      0.97       958
               7       0.97      0.96      0.96      1028
               8       0.96      0.96      0.96       974
               9       0.97      0.95      0.96      1009
    
        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000
    
    



```python

```
