import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer

# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers.dnn import Pool2DDNNLayer as MaxPoolLayer

from lasagnekit.easy import LightweightModel

relu = lasagne.nonlinearities.rectify


def build_model(input_width=32, input_height=32, output_dim=10,
                batch_size=None):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )

    l_conv1 = ConvLayer(
        l_in,
        num_filters=32,
        pad=2,
        filter_size=(5, 5),
        nonlinearity=relu,
    )
    l_pool1 = MaxPoolLayer(
        l_conv1,
        pool_size=(2, 2),
    )
    l_conv2 = ConvLayer(
        l_pool1,
        num_filters=64,
        pad=2,
        filter_size=(5, 5),
        nonlinearity=relu,
    )
    l_pool2 = MaxPoolLayer(
        l_conv2,
        pool_size=(2, 2),
    )

    l_conv3 = ConvLayer(
        l_pool2,
        num_filters=64,
        pad=2,
        filter_size=(5, 5),
        nonlinearity=relu,
    )
    l_pool3 = MaxPoolLayer(
        l_conv3,
        pool_size=(2, 2),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=1024,
        nonlinearity=relu,
    )
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])
