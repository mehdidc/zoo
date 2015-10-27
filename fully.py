from lasagne.layers import DenseLayer, InputLayer
from lasagnekit.easy import LightweightModel
from lasagne.nonlinearities import softmax, rectify


def build_model(input_width=32, input_height=32, output_dim=10):
    net = {}
    net['input'] = InputLayer((None, 3, input_height, input_width))

    net['hidden'] = DenseLayer(net['input'],
                               num_units=100, nonlinearity=rectify)
    net['output'] = DenseLayer(net['hidden'],
                               num_units=output_dim, nonlinearity=softmax)
    return LightweightModel([net['input']], [net['output']])
