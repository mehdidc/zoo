from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagnekit.easy import LightweightModel, BatchOptimizer, linearize
from lasagne.nonlinearities import softmax

def build_model(input_width=32, input_height=32, output_dim=10):
    net = {}
    net['input'] = InputLayer((None, 3, input_height, input_width))
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2)
    net['cccp1'] = ConvLayer(net['conv1'], num_filters=160, filter_size=1)
    net['cccp2'] = ConvLayer(net['cccp1'], num_filters=96, filter_size=1)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2)
    net['cccp3'] = ConvLayer(net['conv2'], num_filters=192, filter_size=1)
    net['cccp4'] = ConvLayer(net['cccp3'], num_filters=192, filter_size=1)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1)
    net['cccp5'] = ConvLayer(net['conv3'], num_filters=192, filter_size=1)
    net['cccp6'] = ConvLayer(net['cccp5'], num_filters=output_dim, filter_size=1)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=8,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = FlattenLayer(net['pool3'])
    net['output'] = NonlinearityLayer(net['output'], softmax)
    return LightweightModel([net['input']], [net['output']])
