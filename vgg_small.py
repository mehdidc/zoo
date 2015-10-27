import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagnekit.easy import LightweightModel

from theano.sandbox import rng_mrg

relu = lasagne.nonlinearities.rectify


def build_model(input_width=32, input_height=32, output_dim=10,
                batch_size=None):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )

    l_conv1 = ConvLayer(
        l_in,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )
    l_conv2 = ConvLayer(
        l_conv1,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )
    l_pool1 = MaxPoolLayer(
        l_conv2,
        pool_size=(2, 2),
    )

    l_pool1 = lasagne.layers.DropoutLayer(l_pool1, 0.25)

    l_conv3 = ConvLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=relu,
    )
    l_conv4 = ConvLayer(
        l_conv3,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=relu,
    )

    l_pool2 = MaxPoolLayer(
        l_conv4,
        pool_size=(2, 2),
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=4096,
        nonlinearity=relu,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=4096,
        nonlinearity=relu,
    )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])
