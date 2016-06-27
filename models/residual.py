import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
# from lasagne.layers import DropoutLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer, DropoutLayer
from lasagnekit.easy import LightweightModel
from hp_toolkit.hp import Param, make_constant_param

relu = lasagne.nonlinearities.very_leaky_rectify


def residual_group(layer, nb=2,
                   num_filters=64,
                   filter_size=3,
                   pad=1,
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
            nonlinearity=None,
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

params = dict(
    f0=Param(initial=16, interval=[16, 32, 64], type='choice'),
    f1=Param(initial=16, interval=[16, 32, 64], type='choice'),
    f2=Param(initial=32, interval=[16, 32, 64], type='choice'),
    f3=Param(initial=64, interval=[16, 32, 64], type='choice'),
    fs0=make_constant_param(3),
    fs1=make_constant_param(3),
    fs2=make_constant_param(3),
    fs3=make_constant_param(3),
    pg1=make_constant_param(2), nbg1=make_constant_param(2),
    pg2=make_constant_param(2), nbg2=make_constant_param(2),
    pg3=make_constant_param(2), nbg3=make_constant_param(2),
    nonlin=Param(initial='rectify',
                 interval=['rectify', 'leaky_rectify', 'very_leaky_rectify'],
                 type='choice'),
)


def build_model(input_width=32, input_height=32, output_dim=10,
                batch_size=None,
                rng=None,
                **hp):
    """
    Residual neural net : http://arxiv.org/pdf/1512.03385v1.pdf
    """
    relu = getattr(lasagne.nonlinearities, hp["nonlin"])

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )
    l_conv = l_in
    l_conv = ConvLayer(
        l_conv,
        num_filters=hp["f0"],
        filter_size=hp["fs0"],
        nonlinearity=relu,
        pad=1,
    )
    print(l_conv.output_shape)
    l_conv = residual_block(
        l_conv, per_group=hp["pg1"], nb_groups=hp["nbg1"],
        filter_size=hp["fs1"],
        num_filters=hp["f1"],
        stride=2)
    print(l_conv.output_shape)
    l_conv = residual_block(
        l_conv, per_group=hp["pg2"], nb_groups=hp["nbg2"],
        filter_size=hp["fs2"],
        num_filters=hp["f2"],
        stride=2)
    print(l_conv.output_shape)
    l_conv = residual_block(
        l_conv, per_group=hp["pg3"], nb_groups=hp["nbg3"],
        filter_size=hp["fs3"],
        num_filters=hp["f3"],
        stride=1)
    print(l_conv.output_shape)
    l_conv = GlobalPoolLayer(l_conv)
    l_out = DenseLayer(
        l_conv,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])
