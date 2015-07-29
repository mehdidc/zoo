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


import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

from datetime import datetime

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
                batch_size=None, 
                dimshuffle=True,
                rng=None):
        
    if rng is None:
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
    l_pool1._srng = rng

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
    l_pool2._srng = rng

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
    l_pool3._srng = rng

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=2048,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.7)
    l_hidden1._srng = rng

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=2048,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2 = lasagne.layers.DropoutLayer(l_hidden2, p=0.7)
    l_hidden2._srng = rng

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )

    return LightweightModel([l_in], [l_out])


def build_model_small(input_width=32, input_height=32, output_dim=10,
                      batch_size=None, dimshuffle=True):

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


def build_model_very_small(input_width=32, input_height=32, output_dim=10,
                           batch_size=None, dimshuffle=True):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
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
from realtime_augmentation import random_perturbation_transform, fast_warp

def Transform(X, rng, zoom_range=None, rotation_range=None, shear_range=None, translation_range=None, do_flip=True):
    if zoom_range is None:
        zoom_range = (1.0, 1.1)
    if rotation_range is None:
        rotation_range = (0, 180)
    if shear_range is None:
        shear_range = (0, 0)
    if translation_range is None:
        translation_range = (-4, 4)
    transf = random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=do_flip)
    X_trans = np.zeros(X.shape, dtype="float32")
    for i in range(X.shape[0]):
       X_trans[i] = fast_warp(X[i], transf, output_shape=(32, 32))
    return X_trans


from lasagne.easy import BatchIterator, get_batch_slice
class MyBatchIterator(BatchIterator):

    def __init__(self, nb_data_augmentation=1,  **transform_params):
        super(MyBatchIterator, self).__init__()

        self.nb_data_augmentation = nb_data_augmentation
        self.transform_params = transform_params

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

        X_list = [X]
        y_list = [y]
        for i in range(self.nb_data_augmentation):
            X_transformed = Transform(X.transpose(0, 3, 2, 1), np.random, **self.transform_params).transpose((0, 3, 2, 1))
            X_list.append(X_transformed)
            y_list.append(y)

        d["X"] = np.concatenate(X_list, axis=0)
        d["y"] = np.concatenate(y_list, axis=0)
        d["X"], d["y"] = shuffle(d["X"], d["y"])
        return d

if __name__ == "__main__":
    from lasagne.datasets.cifar100 import Cifar100
    from lasagne.datasets.cifar10 import Cifar10
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
    from collections import OrderedDict
    import time
    import os
    
    from lightexperiments.light import Light
    from theano.sandbox import rng_mrg

    seed = 1234
    np.random.seed(seed)
    rng = rng_mrg.MRG_RandomStreams(seed)

    light = Light()
    light.launch()
    light.initials()
    light.file_snapshot()
    light.set_seed(seed)
    light.tag("cifar10_vgg")

    hp = dict(
            learning_rate=0.01,
            learning_rate_decay=5.0e-10,
            weight_decay=0.00002,
            max_nb_epochs=120,
            batch_size=64,
            momentum=0.9,
            patience_nb_epochs=800,
            patience_threshold=1.14,
            patience_check_each=5,

            # data augmentation
            nb_data_augmentation=2,
            zoom_range=(1.0, 1.3),
            rotation_range=(0, 360),
            shear_range=(0, 0),
            translation_range=(-2, 2),
            do_flip=True

    )
    for k, v in hp.items():
        light.set(k, v)

    use_pylearn_data = True
    use_bokeh = False

    if use_bokeh:
        from bokeh.plotting import cursession, figure, show, output_server
        output_server("cifar10", url="http://onevm-60.lal.in2p3.fr:15000")
        print(cursession().__dict__)
        p = figure()
        p.line([], [], name="learning_curve_train", legend="learning curve train", color="blue")
        p.line([], [], name="learning_curve_valid", legend="learning curve valid", color="green")
        show(p)
        renderer = p.select(dict(name="learning_curve_train"))
        curve_train_ds = renderer[0].data_source
        renderer = p.select(dict(name="learning_curve_valid"))
        curve_valid_ds = renderer[0].data_source

    model = build_model(input_width=32, input_height=32,
                        output_dim=10,
                        rng=rng)
    

    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            start = datetime.now()
            status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            duration = (datetime.now() - start).total_seconds()
            status["duration"] = duration

            s = np.arange(X_train.shape[0])
            np.random.shuffle(s)
            s = s[0:1000]
            status["accuracy_train"] = (nnet.predict(X_train[s])==y_train[s]).mean()
            s = np.arange(X_test.shape[0])
            np.random.shuffle(s)
            s = s[0:1000]
            status["accuracy_valid"] = (nnet.predict(X_test[s])==y_test[s]).mean()
            status["error_valid"] = 1 - status["accuracy_valid"]
            
            if use_bokeh:
                curve_train_ds.data["x"].append(status["epoch"])
                curve_valid_ds.data["x"].append(status["epoch"])
                curve_train_ds.data["y"].append(1-status["accuracy_train"])
                curve_valid_ds.data["y"].append(1-status["accuracy_valid"])
                cursession().store_objects(curve_train_ds)
                cursession().store_objects(curve_valid_ds)
            decay = np.array((1 - hp["learning_rate_decay"]), dtype="float32")
            self.learning_rate.set_value(self.learning_rate.get_value() * decay)

            for k, v in status.items():
                light.append(k, float(v))
            light.append("learning_rate_per_epoch", float(self.learning_rate.get_value()))
            return status

    learning_rate = theano.shared(np.array(hp["learning_rate"], dtype="float32"))
    batch_optimizer = MyBatchOptimizer(
        verbose=2, max_nb_epochs=hp["max_nb_epochs"],
        batch_size=hp["batch_size"],
        optimization_procedure=(updates.momentum, {"learning_rate": learning_rate, "momentum": hp["momentum"]}),
        #whole_dataset_in_device=True,
        patience_stat="error_valid",
        patience_nb_epochs=hp["patience_nb_epochs"],
        patience_progression_rate_threshold=hp["patience_threshold"],
        patience_check_each=hp["patience_check_each"],
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
        if hp["weight_decay"] > 0:
            l1 = sum(T.abs_(param).sum() for param in model.capsule.all_params_regularizable) * hp["weight_decay"]
        else:
            l1 = 0
        return T.nnet.categorical_crossentropy(y_hat, y).mean() + l1

    batch_iterator = MyBatchIterator(hp["nb_data_augmentation"],
                                     zoom_range=hp["zoom_range"],
                                     rotation_range=hp["rotation_range"],
                                     shear_range=hp["shear_range"],
                                     translation_range=hp["translation_range"],
                                     do_flip=hp["do_flip"])

    nnet = Capsule(
        input_variables, model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=batch_iterator,
    )
    
    if use_pylearn_data is True:
        directory = "{0}/cifar10/pylearn2_gcn_whitened/".format(os.getenv("DATA_PATH"))
        import cPickle as pickle
        train = pickle.load(open(directory + "train.pkl"))
        test = pickle.load(open(directory + "test.pkl"))
        X_train = train.X.reshape((train.X.shape[0], 3, 32, 32))
        X_test = test.X.reshape((test.X.shape[0], 3, 32, 32))
        y_train = train.y[:, 0]
        y_test = test.y[:, 0]
    
    elif not os.path.exists("data_cached.npz"):
        data = Cifar10(batch_indexes=[1, 2, 3, 4, 5, 6])
        data.load()
        imshape = (data.X.shape[0], 3, 32, 32)
        
        X = data.X.reshape(imshape).astype(np.float32) / 255.
        y = data.y
        y = y.astype(np.int32)

        X, y = shuffle(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        pipeline = make_pipeline(MinMaxScaler())
        pipeline.fit(linearize(X_train))

        X_train = pipeline.transform(linearize(X_train)).reshape((X_train.shape[0], 3, 32, 32))
        X_test = pipeline.transform(linearize(X_test)).reshape((X_test.shape[0], 3, 32, 32))

        np.savez("data_cached.npz", 
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test)
    else:
        data = np.load("data_cached.npz")
        X_train, y_train, X_test, y_test = (data["X_train"],
                                           data["y_train"],
                                           data["X_test"],
                                           data["y_test"])
        print(X_train.max())
    try:
        nnet.fit(X=X_train, y=y_train)
    except KeyboardInterrupt:
        print("interruption...")
    light.endings() # save the duration
    light.store_experiment() # update the DB
    light.close()
