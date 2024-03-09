from neural_net.architecture import Sequential
from neural_net.layers import Fullyconnected,Activation
from neural_net.init_funcs import zeros
from neural_net.activation import σ,Softmax
from neural_net.cost import BinaryCrossEntropy,CrossEntropy
from neural_net.utils import make_circle_data
from neural_net.pipeline import onehot
import numpy


softmax = Sequential(
        [
        Fullyconnected(2,2,zeros),
        Activation(Softmax)
        ],
    CrossEntropy
    )

if __name__ == "__main__":
    centers = [(-50, 0), (20, 30)]
    radii = [30, 35]
    X, y = make_circle_data(centers, radii)
    y = onehot(y)


softmax.train(X,y,epochs=1000,α=0.1)