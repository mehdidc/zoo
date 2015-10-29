import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagne.layers import DropoutLayer
from lasagnekit.easy import LightweightModel

from theano.sandbox import rng_mrg

#relu = lasagne.nonlinearities.rectify
relu = lasagne.nonlinearities.very_leaky_rectify

def build_model(input_width=32, input_height=32, output_dim=10,
                l=5, k=100,dropout=True,
                batch_size=None):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )
    l_conv = l_in
    for i in range(l):
        if i == 0:
            pad = (input_height*3, input_width*3)
            filter_size = (3, 3)
        else:
            pad = 0
            filter_size = (2, 2)
        l_conv = ConvLayer(
            l_conv,
            num_filters=(i + 1) * k,
            filter_size=filter_size,
            pad=pad,
            nonlinearity=relu,
        )
        if dropout is True:
            p = min(i / 10., 0.5)
            if p > 0:
                l_conv = DropoutLayer(l_conv, p=p)

        l_pool = MaxPoolLayer(
            l_conv,
            pool_size=(2, 2),
        )
        l_conv = l_pool
    l_conv = ConvLayer(
        l_conv,
        num_filters=(i + 1) * k,
        filter_size=filter_size,
        pad=0,
        nonlinearity=relu
    )
    if dropout is True:
        p = min(i / 10., 0.5)
        if p > 0:
            l_conv = DropoutLayer(l_conv, p=p)

    l_out = lasagne.layers.DenseLayer(
        l_conv,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])
