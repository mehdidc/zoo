import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagnekit.easy import LightweightModel

from theano.sandbox import rng_mrg
from hp_toolkit.hp import Param

params = dict(
    f1=Param(initial=32, interval=[2, 256], type='int'),
    f2=Param(initial=64, interval=[2, 256], type='int'),
    f3=Param(initial=128, interval=[2, 256], type='int'),
    ds1=Param(initial=2048, interval=[64, 4096], type='int'),
    ds2=Param(initial=2048, interval=[64, 4096], type='int'),
    do1=Param(initial=0.25, interval=[0, 1], type='real'),
    do2=Param(initial=0.25, interval=[0, 1], type='real'),
    do3=Param(initial=0.25, interval=[0, 1], type='real'),
    do4=Param(initial=0.7, interval=[0, 1], type='real'),
    do5=Param(initial=0.7, interval=[0, 1], type='real'),
    nonlin=Param(initial='rectify',
                 interval=['rectify', 'leaky_rectify', 'very_leaky_rectify'],
                 type='choice'),
)


def build_model(input_width=224, input_height=224, output_dim=1000,
                batch_size=None,
                rng=None,
                **hp):

    if rng is None:
        rng = rng_mrg.MRG_RandomStreams(1234)

    relu = getattr(lasagne.nonlinearities, hp["nonlin"])

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )

    l_conv1 = ConvLayer(
        l_in,
        num_filters=hp["f1"] * 2,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )
    l_conv2 = ConvLayer(
        l_conv1,
        num_filters=hp["f1"] * 2,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )
    l_pool1 = MaxPoolLayer(
        l_conv2,
        pool_size=(2, 2),
    )

    l_pool1 = lasagne.layers.DropoutLayer(l_pool1, p=hp["do1"])
    l_pool1._srng = rng

    l_conv3 = ConvLayer(
        l_pool1,
        num_filters=hp["f2"] * 2,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )
    l_conv4 = ConvLayer(
        l_conv3,
        num_filters=hp["f2"] * 2,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )

    l_pool2 = MaxPoolLayer(
        l_conv4,
        pool_size=(2, 2),
    )

    l_pool2 = lasagne.layers.DropoutLayer(l_pool2, p=hp["do2"])
    l_pool2._srng = rng

    l_conv5 = ConvLayer(
        l_pool2,
        pad=1,
        num_filters=hp["f3"] * 2,
        filter_size=(3, 3),
        nonlinearity=relu,
    )
    l_conv6 = ConvLayer(
        l_conv5,
        num_filters=hp["f3"] * 2,
        pad=1,
        filter_size=(3, 3),
        nonlinearity=relu,
    )

    l_conv7 = ConvLayer(
        l_conv6,
        num_filters=hp["f3"] * 2,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=relu,
    )

    l_conv8 = ConvLayer(
        l_conv7,
        num_filters=hp["f3"] * 2,
        pad=1,
        filter_size=(3, 3),
        nonlinearity=relu,
    )

    l_pool3 = MaxPoolLayer(
        l_conv8,
        pool_size=(2, 2),
    )

    l_pool3 = lasagne.layers.DropoutLayer(l_pool3, p=hp["do3"])
    l_pool3._srng = rng

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=hp["ds1"],
        nonlinearity=relu,
    )
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=hp["do4"])
    l_hidden1._srng = rng

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=hp["ds2"],
        nonlinearity=relu,
    )
    l_hidden2 = lasagne.layers.DropoutLayer(l_hidden2, p=hp["do5"])
    l_hidden2._srng = rng

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )

    return LightweightModel([l_in], [l_out])
