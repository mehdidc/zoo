import lasagne
from lasagne.layers import cuda_convnet

BATCH_SIZE = 256


def build_model(input_width=224, input_height=224, output_dim=1000,
                batch_size=BATCH_SIZE, dimshuffle=True):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_width, input_height),
    )

    if not dimshuffle:
        l_in = cuda_convnet.bc01_to_c01b(l_in)

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=64,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool1 = cuda_convnet.MaxPool2DCCLayer(
        l_conv1,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool2 = cuda_convnet.c01b_to_bc01(l_pool2)

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool2,
        num_filters=256,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv4 = cuda_convnet.Conv2DCCLayer(
        l_conv3,
        num_filters=256,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv4,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool3 = cuda_convnet.c01b_to_bc01(l_pool3)

    l_conv5 = cuda_convnet.Conv2DCCLayer(
        l_pool3,
        num_filters=512,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv6 = cuda_convnet.Conv2DCCLayer(
        l_conv5,
        num_filters=512,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_pool4 = cuda_convnet.MaxPool2DCCLayer(
        l_conv6,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool4 = cuda_convnet.c01b_to_bc01(l_pool4)


    l_conv7 = cuda_convnet.Conv2DCCLayer(
        l_pool4,
        num_filters=512,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv8 = cuda_convnet.Conv2DCCLayer(
        l_conv7,
        num_filters=512,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_pool5 = cuda_convnet.MaxPool2DCCLayer(
        l_conv8,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool5 = cuda_convnet.c01b_to_bc01(l_pool5)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool5,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out


