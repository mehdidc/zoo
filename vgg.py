import lasagne
from lasagne.layers import cuda_convnet
from lasagne.easy import LightweightModel, BatchOptimizer
from lasagne.generative.capsule import Capsule
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

    #l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    l_hidden1_dropout = l_hidden1
    #l_hidden1_dropout = l_hidden1

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = l_hidden2
     #l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
     #l_hidden2_dropout = l_hidden1_dropout

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return LightweightModel([l_in], [l_out])

# source : hTps://github.com/rakeshvar/theanet/blob/master/theanet/layer/inlayers.py
import theano.tensor.signal.conv as sigconv

def image_transform(inpt, img_sz,
                    rng,
                    translation=0,
                    zoom=1,
                    magnitude=0,
                    sigma=1,
                    pflip=0,
                    angle=0,
                    rand_gen=None,
                    invert_image=False,
                    nearest=True):
    assert zoom > 0
    h = w = img_sz

    # Humble as-is beginning
    target = T.as_tensor_variable(np.indices((h, w)))

    # Translate
    if translation:
        transln = translation * rng.uniform((2, 1, 1), -1)
        target += transln

    # Apply elastic transform
    if magnitude:
        # Build a gaussian filter
        var = sigma ** 2
        filt = np.array([[np.exp(-.5 * (i * i + j * j) / var)
                            for i in range(-sigma, sigma + 1)]
                            for j in range(-sigma, sigma + 1)], dtype=theano.config.floatX)
        filt /= 2 * np.pi * var

        # Elastic
        elast = magnitude * rng.normal((2, h, w))
        elast = sigconv.conv2d(elast, filt, (2, h, w), filt.shape, 'full')
        elast = elast[:, sigma:h + sigma, sigma:w + sigma]
        target += elast

    # Center at 'about' half way
    if zoom-1 or angle:
        origin = rng.uniform((2, 1, 1), .25, .75) * \
                    np.array((h, w)).reshape((2, 1, 1))
        target -= origin

        # Zoom
        if zoom-1:
            zoomer = T.exp(np.log(zoom) * rng.uniform((2, 1, 1), -1))
            target *= zoomer

        # Rotate
        if angle:
            theta = angle * np.pi / 180 * rng.uniform(low=-1)
            c, s = T.cos(theta), T.sin(theta)
            rotate = T.stack(c, -s, s, c).reshape((2,2))
            target = T.tensordot(rotate, target, axes=((0, 0)))

        # Uncenter
        target += origin

    # Clip the mapping to valid range and linearly interpolate
    transy = T.clip(target[0], 0, h - 1 - .001)
    transx = T.clip(target[1], 0, w - 1 - .001)

    if nearest:
        vert = T.iround(transy)
        horz = T.iround(transx)
        output = inpt[:, :, vert, horz]
    else:
        topp = T.cast(transy, 'int32')
        left = T.cast(transx, 'int32')
        fraction_y = T.cast(transy - topp, theano.config.floatX)
        fraction_x = T.cast(transx - left, theano.config.floatX)

        output = inpt[:, :, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                inpt[:, :, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                inpt[:, :, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                inpt[:, :, topp + 1, left + 1] * fraction_y * fraction_x

    # Now add some noise
    if pflip:
        mask = rng.binomial(n=1, p=pflip, size=inpt.shape, dtype=theano.config.floatX)
        output = (1 - output) * mask + output * (1 - mask)
    return output

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
        X_transformed = V["X"][batch_slice]
        y_transformed = V["y"][batch_slice]
        X_transformed = image_transform(X_transformed, 32, self.model.rng,
                                        translation=1,
                                        zoom=1,
                                        magnitude=1,
                                        sigma=1,
                                        pflip=1,
                                        angle=10)
        d["X"] = X_transformed
        d["y"] = y_transformed
        return d

if __name__ == "__main__":
    from lasagne.datasets.cifar100 import Cifar100
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    import numpy as np
    from collections import OrderedDict
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

    batch_optimizer = MyBatchOptimizer(
        verbose=1, max_nb_epochs=100,
        batch_size=BATCH_SIZE,
        optimization_procedure=(updates.adagrad, {"learning_rate":1e-2}),
        whole_dataset_in_device=True
    )

    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor4)
    input_variables["y"] = dict(tensor_type=T.ivector)
    functions = dict(
        predict=dict(
            get_output=lambda model, X: (model.get_output(X)[0]).argmax(axis=1),
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

    data = Cifar100(which='all')
    data.load()
    X = data.X.reshape((data.X.shape[0], 3, 32, 32))
    y = data.y
    y = y.astype(np.int32)
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    nnet.fit(X=X_train, y=y_train)
