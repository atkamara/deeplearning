from neural_net import *

init_method =lambda _in,out: utils.numpy.random.uniform(-1./out**.5, 1/out**.5, (_in+1,1))


ann = architecture.Sequential(
        [
        layers.fullyconnected(n_in=2,n_out=50,init_method=init_method,store=False),
        layers.activation(n_in=50,n_out=50,func=activation_funcs.LeakyReLU,store=False),
        
            
        layers.fullyconnected(n_in=50,n_out=2,init_method=init_method,store=False),
        layers.activation(n_in=2,n_out=1,func=activation_funcs.Softmax,store=False)
        ],
    cost_func= loss.CrossEntropy
    )
X = utils.numpy.random.uniform(-1,1,size=(100,2))
y = (X.sum(axis=1) < utils.numpy.random.uniform(.3,.37,(len(X),))).reshape(-1,1)+0


if __name__ == '__main__':

    ann.eval(X)
    ann.train(X,y,n_epochs=20,metrics=metrics.accuracy)