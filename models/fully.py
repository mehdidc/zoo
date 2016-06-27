from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagnekit.easy import LightweightModel
from lasagne.nonlinearities import softmax, rectify

params = {}


def build_model(input_width=32, input_height=32, output_dim=10, **hp):
    net = {}
    net['input'] = InputLayer((None, 3, input_height, input_width))

    net['hidden'] = DenseLayer(net['input'],
                               num_units=1000, nonlinearity=rectify)
    net['hidden'] = DropoutLayer(net['hidden'], p=0.5)
    net['hidden2'] = DenseLayer(net['hidden'],
                                num_units=1000, nonlinearity=rectify)

    net['hidden2'] = DropoutLayer(net['hidden2'], p=0.5)
    net['output'] = DenseLayer(net['hidden2'],
                               num_units=output_dim, nonlinearity=softmax)
    return LightweightModel([net['input']], [net['output']])
