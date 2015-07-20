import matplotlib as mpl
mpl.use('Agg')
import lasagne
from lasagne.layers import cuda_convnet
from lasagne.easy import LightweightModel, BatchOptimizer, linearize
from lasagne.generative.capsule import Capsule
from lasagne import updates
import theano
import theano.tensor as T
from theano.sandbox import rng_mrg

BATCH_SIZE = 64


import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

# source : https://gist.github.com/duschendestroyer/5170087

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = linearize(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = linearize(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

def build_model(input_width=224, input_height=224, output_dim=1000,
                batch_size=BATCH_SIZE, dimshuffle=True,
                rng=None):
    
    rng = rng_mrg.MRG_RandomStreams(1234)

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )

    if not dimshuffle:
        l_in = cuda_convnet.bc01_to_c01b(l_in)

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool1 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool1 = cuda_convnet.c01b_to_bc01(l_pool1)

    l_pool1 = lasagne.layers.DropoutLayer(l_pool1, p=0.25)

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv4 = cuda_convnet.Conv2DCCLayer(
        l_conv3,
        num_filters=128,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv4,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool2 = cuda_convnet.c01b_to_bc01(l_pool2)

    l_pool2 = lasagne.layers.DropoutLayer(l_pool2, p=0.25)

    l_conv5 = cuda_convnet.Conv2DCCLayer(
        l_pool2,
        pad=1,
        num_filters=256,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv6 = cuda_convnet.Conv2DCCLayer(
        l_conv5,
        num_filters=256,
        pad=1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_conv7 = cuda_convnet.Conv2DCCLayer(
        l_conv6,
        num_filters=256,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )

    l_conv8 = cuda_convnet.Conv2DCCLayer(
        l_conv7,
        num_filters=256,
        pad=1,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )


    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv8,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool3 = cuda_convnet.c01b_to_bc01(l_pool3)
    l_pool3 = lasagne.layers.DropoutLayer(l_pool3, p=0.25)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=1024,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=1024,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2 = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )

    return LightweightModel([l_in], [l_out])


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
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_conv1,
        num_filters=64,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool1 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_pool2 = cuda_convnet.c01b_to_bc01(l_pool2)
    
    l_pool1 = lasagne.layers.DropoutLayer(l_pool1, 0.25)

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_conv4 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=128,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv4,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )
    if not dimshuffle:
        l_pool2 = cuda_convnet.c01b_to_bc01(l_pool3)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )


    l_hidden1_dropout = l_hidden1
    #l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = l_hidden2
    #l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])


def build_model_very_small(input_width=32, input_height=32, output_dim=100,
                     batch_size=BATCH_SIZE, dimshuffle=True):

    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, input_width, input_height),
    )

    if not dimshuffle:
        l_in = cuda_convnet.bc01_to_c01b(l_in)

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=32,
        pad=2,
        filter_size=(5, 5),
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
        num_filters=64,
        pad=2,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=64,
        pad=2,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )
    if not dimshuffle:
        l_pool3 = cuda_convnet.c01b_to_bc01(l_pool3)
    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=1024,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])


from skimage.transform import AffineTransform
import numpy as np
from skimage import transform as tf
def Transform(X, rng):
    translation = rng.randint(-2, 2, size=2)
    scale = rng.uniform(1, 1.4, size=2)
    rotation = rng.uniform(-np.pi/6, np.pi/6)
    affine_transform = AffineTransform(translation=translation, scale=scale, rotation=rotation)
    X_trans = np.zeros(X.shape, dtype="float32")
    for i in range(X.shape[0]):
       X_trans[i] = tf.warp(X[i], affine_transform)
    return X_trans


from lasagne.easy import BatchIterator, get_batch_slice
class MyBatchIterator(BatchIterator):

     def transform(self, batch_index, V):
        assert self.batch_size is not None
        assert self.nb_batches is not None

        if isinstance(batch_index, T.TensorVariable):
            batch_slice = get_batch_slice(batch_index,
                                          self.batch_size)
        else:
            batch_slice = slice(batch_index * self.batch_size,
                                (batch_index+1) * self.batch_size)

        d = OrderedDict()
        X = V["X"][batch_slice]
        y = V["y"][batch_slice]
        #X_transformed = Transform(X.transpose(0, 3, 2, 1), np.random).transpose((0, 3, 2, 1))
        #d["X"] = np.concatenate([X, X_transformed], axis=0)
        #d["y"] = np.concatenate([y, y], axis=0)
        d["X"] = X
        d["y"] = y
        return d

if __name__ == "__main__":
    from lasagne.datasets.cifar100 import Cifar100
    from lasagne.datasets.cifar10 import Cifar10
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    import numpy as np
    from collections import OrderedDict
    import time

    from bokeh.plotting import cursession, figure, show, output_server
    output_server("cifar10", url="http://onevm-60.lal.in2p3.fr:15000")
    p = figure()
    p.line(np.arange(0, 100), np.arange(0, 100)*2, name="learning_curve_train")
    p.line(np.arange(0, 100), np.arange(0, 100)*2, name="learning_curve_valid")
    show(p)
    renderer = p.select(dict(name="learning_curve_train"))
    curve_train_ds = renderer[0].data_source
    renderer = p.select(dict(name="learning_curve_valid"))
    curve_valid_ds = renderer[0].data_source

    model = build_model(input_width=32, input_height=32,
                        output_dim=10)

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
            status["error_valid"] = 1 - status["accuracy_valid"]
            #curve_train_ds.data["x"].append(status["epoch"])
            #curve_valid_ds.data["x"].append(status["epoch"])
            #curve_train_ds.data["y"].append(status["accuracy_train"])
            #curve_valid_ds.data["y"].append(status["accuracy_valid"])
            cursession().store_objects(curve_train_ds)
            cursession().store_objects(curve_valid_ds)
            decay = np.array((1 - 5.0e-6), dtype="float32")
            self.learning_rate.set_value(self.learning_rate.get_value() * decay)
            return status


    learning_rate = theano.shared(np.array(0.01, dtype="float32"))
    batch_optimizer = MyBatchOptimizer(
        verbose=2, max_nb_epochs=1000,
        batch_size=BATCH_SIZE,
        optimization_procedure=(updates.momentum, {"learning_rate": learning_rate, "momentum": 0.9}),
        whole_dataset_in_device=True,
        patience_stat="error_valid",
        patience_nb_epochs=20,
        patience_progression_rate_threshold=1.14,
        patience_check_each=5,
    )
    batch_optimizer.learning_rate = learning_rate

    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor4)
    input_variables["y"] = dict(tensor_type=T.ivector)
    functions = dict(
        predict=dict(
            get_output=lambda model, X: (model.get_output(X, deterministic=True)[0]).argmax(axis=1),
            params=["X"]
        )
    )

    def loss_function(model, tensors):
        X = tensors["X"]
        y = tensors["y"]
        y_hat, = model.get_output(X)
        return T.nnet.categorical_crossentropy(y_hat, y).mean()

    nnet = Capsule(
        input_variables, model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=MyBatchIterator(),
    )

    data = Cifar10(batch_indexes=[1, 2, 3, 4, 5, 6])
    data.load()
    imshape = (data.X.shape[0], 3, 32, 32)

    X = data.X.reshape(imshape).astype(np.float32)
    y = data.y
    y = y.astype(np.int32)

    X, y = shuffle(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    zca = ZCA()
    zca.fit(X_train)
    X_train = zca.transform(X_train).reshape((X_train.shape[0], 3, 32, 32))
    X_test = zca.transform(X_test).reshape((X_test.shape[0], 3, 32, 32))

    nnet.fit(X=X_train, y=y_train)
