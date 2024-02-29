from src import *
import numpy





ann_sigmoid = architecture.Sequential(
     [

       layers.Fullyconnected(n_in=2,n_out=50,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.σ),
       layers.Fullyconnected(n_in=50,n_out=1,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.σ),


    ],
    cost = cost.binaryCrossEntropy
)
ann_softmax = architecture.Sequential(
     [

       layers.Fullyconnected(n_in=2,n_out=50,init_method=init_funcs.XHReluuniform) ,
       layers.Activation(activation.LeakyReLU),
       layers.Fullyconnected(n_in=50,n_out=2,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.Softmax),


    ],
    cost = cost.CrossEntropy
)
ann_mse = architecture.Sequential(
     [

       layers.Fullyconnected(n_in=2,n_out=50,init_method=init_funcs.XHReluuniform) ,
       layers.Activation(activation.LeakyReLU),
       layers.Fullyconnected(n_in=50,n_out=2,init_method=init_funcs.XHsigmoiduniform) ,
       layers.Activation(activation.Softmax),


    ],
    cost = cost.MSE
)

if __name__ == '__main__':
    n,k = 5000,2
    X = numpy.random.uniform(-100,100,size=(n,k))
    y =( (X[:, 0]**2 + X[:, 1]**2)/numpy.pi < 1000).reshape(-1,1)+0


    X,y = pipeline.shuffle(X,y)

    X = pipeline.scaler(X)

    batch = pipeline.Batch(50,n, lambda : X, lambda : y)
    
    ann_sigmoid.train(batch=batch,α=.5,epochs=20,metrics=metrics.accuracy)

    batch = pipeline.Batch(50,n, lambda : X, lambda : pipeline.onehot(y))
    ann_softmax.train(batch=batch,α=.5,epochs=20,metrics=metrics.accuracy)

    ann_mse.train(batch=batch,α=.5,epochs=20,metrics=metrics.accuracy)
