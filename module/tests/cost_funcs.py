from neural_net.cost import *

def test_cost_value():
    y_true = numpy.array([[0], [1], [1], [0]])
    predicted_probs = numpy.array([[0.2], [0.8], [0.6], [0.3]])
    bce_loss = binaryCrossEntropy()
    loss_value = bce_loss.compute(y_true, predicted_probs)
    assert loss_value == 0.3284
def test_cost_derivative():
    y_true = numpy.array([[0], [1], [1], [0]])
    predicted_probs = numpy.array([[0.2], [0.8], [0.6], [0.3]])
    bce_loss = binaryCrossEntropy()
    derivative_values = bce_loss.pr()
    assert derivative_values == [ 1.25,-1.25,-1.66666667,1.42857143]