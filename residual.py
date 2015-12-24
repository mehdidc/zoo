import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
# from lasagne.layers import DropoutLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer, DropoutLayer
from lasagnekit.easy import LightweightModel

relu = lasagne.nonlinearities.very_leaky_rectify


def residual_group(layer, nb=2,
                   num_filters=64,
                   filter_size=3,
                   pad='same',
                   stride=1,
                   nonlinearity=relu):
    l_conv = layer
    prev_num_filters = layer.output_shape[1]
    if (prev_num_filters != num_filters or
        stride != 1):
        residual = ConvLayer(
            l_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=stride,
            nonlinearity=None
        )
    else:
        residual = l_conv

    for i in range(nb):
        l_conv = ConvLayer(
            l_conv,
            num_filters=num_filters,
            filter_size=(filter_size, filter_size),
            pad=pad,
            nonlinearity=nonlinearity,
            stride=stride
        )
        stride = 1
    return ElemwiseSumLayer([l_conv, residual])


def residual_block(layer, per_group, nb_groups,
                   filter_size=3,
                   num_filters=64,
                   stride=1,
                   nonlinearity=relu):
    for i in range(nb_groups):
        layer = residual_group(layer, nb=per_group,
                               filter_size=filter_size,
                               num_filters=num_filters,
                               stride=stride,
                               nonlinearity=nonlinearity)
        stride = 1
    return layer


def build_model(input_width=32, input_height=32, output_dim=10,
                batch_size=None):
    """
    Residual neural net : http://arxiv.org/pdf/1512.03385v1.pdf
    """
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )
    l_conv = l_in
    l_conv = ConvLayer(
        l_conv,
        num_filters=64,
        filter_size=7,
        nonlinearity=relu
    )
    l_conv = MaxPoolLayer(
        l_conv,
        pool_size=(2, 2)
    )
    l_conv = residual_block(
        l_conv, per_group=2, nb_groups=3,
        filter_size=3,
        num_filters=64)
    print(l_conv.output_shape)
    l_conv = DropoutLayer(l_conv, 0.2)
    l_conv = residual_block(
        l_conv, per_group=2, nb_groups=4,
        filter_size=3,
        num_filters=128,
        stride=2)
    print(l_conv.output_shape)
    l_conv = DropoutLayer(l_conv, 0.2)
    l_conv = residual_block(
        l_conv, per_group=2, nb_groups=3,
        filter_size=3,
        num_filters=128,
        stride=2)
    print(l_conv.output_shape)
    l_conv = DropoutLayer(l_conv, 0.2)
    l_conv = residual_block(
        l_conv, per_group=2, nb_groups=2,
        filter_size=3,
        num_filters=256,
        stride=2)
    #l_conv = DropoutLayer(l_conv, 0.5)
    print(l_conv.output_shape)
    l_conv = GlobalPoolLayer(l_conv)
    l_out = DenseLayer(
        l_conv,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])
