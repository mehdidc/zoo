import lasagne
from lasagne.layers import cuda_convnet
from lasagne.easy import LightweightModel, BatchOptimizer
from lasagne.generative.neural_net import NeuralNet
from lasagne import updates
import theano
import theano.tensor as T
BATCH_SIZE = 256

def build_model(input_width=224, input_height=224, output_dim=1000,
                batch_size=BATCH_SIZE, dimshuffle=True):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
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


def build_model_small(input_width=32, input_height=32, output_dim=100,
                batch_size=BATCH_SIZE, dimshuffle=True):

    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, input_width, input_height),
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
        filter_size=(4, 4),
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
    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv3,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )
    if not dimshuffle:
        l_pool3 = cuda_convnet.c01b_to_bc01(l_pool3)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    #l_hidden1_dropout = l_hidden1

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
     #l_hidden2_dropout = l_hidden1_dropout

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])

if __name__ == "__main__":
    from lasagne.datasets.cifar100 import Cifar100
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    import numpy as np
    model = build_model_small()

    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            s = np.arange(X_train.shape[0])
            np.random.shuffle(s)
            s = s[0:1000]
            status["accuracy_train"] = (nnet.predict(X_train[s])==y_train[s]).mean()
            s = np.arange(X_test.shape[0])
            np.random.shuffle(s)
            s = s[0:1000]
            status["accuracy_valid"] = (nnet.predict(X_test[s])==y_test[s]).mean()
            return status
 
    batch_optimizer = MyBatchOptimizer(verbose=1, max_nb_epochs=100, batch_size=BATCH_SIZE,
            optimization_procedure=(updates.momentum, {"learning_rate":1e-1, "momentum": 0.8}))
    nnet = NeuralNet(model,
                     X_type=T.tensor4,
                     batch_optimizer=batch_optimizer)

    data = Cifar100(which='all')
    data.load()

    X = data.X.reshape((data.X.shape[0], 3, 32, 32))
    y = data.y
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    nnet.fit(X_train, y_train)
