from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
from lasagnekit.easy import BatchOptimizer, BatchIterator, get_batch_slice
from lasagnekit.generative.capsule import Capsule
from lasagnekit.easy import iterate_minibatches
from lasagne import updates
import theano
import theano.tensor as T


import numpy as np


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
       X_trans[i] = fast_warp(X[i], transf, output_shape=(64, 64))
    return X_trans


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
            tr = Transform(X.transpose(0, 3, 2, 1),
                           np.random,
                           **self.transform_params)
            X_transformed = tr.transpose((0, 3, 2, 1))
            X_list.append(X_transformed)
            y_list.append(y)

        d["X"] = np.concatenate(X_list, axis=0)
        d["y"] = np.concatenate(y_list, axis=0)
        d["X"], d["y"] = shuffle(d["X"], d["y"])
        return d

if __name__ == "__main__":
    from lasagnekit.datasets.cifar10 import Cifar10
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    from collections import OrderedDict

    from lightexperiments.light import Light
    from lasagne.random import set_rng
    #from theano.sandbox import rng_mrg

    seed = 1234

    #rng = rng_mrg.MRG_RandomStreams(seed)

    np.random.seed(seed)

    light = Light()
    light.launch()
    light.initials()
    light.file_snapshot()
    light.set_seed(seed)

    data = Cifar10(batch_indexes=[1, 2, 3, 4, 5, 6])
    data.load()

    light.set("dataset", data.__class__.__name__)

    hp = dict(
        learning_rate=0.001,
        learning_rate_decay=5.0e-10,
        weight_decay=0,
        max_nb_epochs=120,
        batch_size=64,
        momentum=0.9,

        patience_nb_epochs=20,
        patience_threshold=1,
        patience_check_each=1,

        # data augmentation
        nb_data_augmentation=0,
        zoom_range=(1.0, 1.3),
        rotation_range=(0, 360),
        shear_range=(0, 0),
        translation_range=(-2, 2),
        do_flip=True

    )

    light.set("hp", hp)

    import vgg # NOQA
    import vgg_small # NOQA
    import vgg_very_small # NOQA
    import spatially_sparse # NOQA
    import nin # NOQA
    import fully # NOQA
    model_class = spatially_sparse
    model = model_class.build_model(
        input_width=data.img_dim[1],
        input_height=data.img_dim[2],
        output_dim=data.output_dim)
    light.set("model", model_class.__name__)

    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            start = datetime.now()
            status = super(MyBatchOptimizer, self).iter_update(epoch,
                                                               nb_batches,
                                                               iter_update_batch)
            duration = (datetime.now() - start).total_seconds()
            status["duration"] = duration
            accs = []
            for mini_batch in iterate_minibatches(X_train.shape[0],
                                                  hp["batch_size"]):
                acc = (nnet.predict(X_train[mini_batch])==y_train[mini_batch])
                accs.append(acc)
            status["accuracy_train"] = np.mean(accs)
            status["accuracy_train_std"] = np.std(accs)

            accs = []
            for mini_batch in iterate_minibatches(X_test.shape[0],
                                                  hp["batch_size"]):
                acc = (nnet.predict(X_test[mini_batch])==y_test[mini_batch])
                accs.append(acc)

            status["accuracy_test"] = np.mean(accs)
            status["accuracy_test_std"] = np.std(accs)

            status["error_valid"] = 1 - status["accuracy_test"]

            for k, v in status.items():
                light.append(k, float(v))
            light.append("learning_rate_per_epoch", float(self.learning_rate.get_value()))
            return status

    learning_rate = theano.shared(np.array(hp["learning_rate"], dtype="float32"))
    batch_optimizer = MyBatchOptimizer(
        verbose=2, max_nb_epochs=hp["max_nb_epochs"],
        batch_size=hp["batch_size"],
        optimization_procedure=(updates.momentum, {"learning_rate": learning_rate}),
        whole_dataset_in_device=True,
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
        #batch_iterator=batch_iterator,
    )

    from sklearn.preprocessing import LabelEncoder

    imshape = ([data.X.shape[0]] +
               list(data.img_dim))
    X = data.X.reshape(imshape).astype(np.float32)
    y = data.y
    y = LabelEncoder().fit_transform(y)
    y = y.astype(np.int32)

    # rescaling to [-1, 1]
    X_min = X.min(axis=(0, 2, 3))[None, :, None, None]
    X_max = X.max(axis=(0, 2, 3))[None, :, None, None]
    X = 2 * ((X - X_min) / (X_max - X_min)) - 1
    X, y = shuffle(X, y)

    X = X[0:10000]
    y = y[0:10000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    light.set("nb_examples_train", X_train.shape[0])
    light.set("nb_examples_test", X_test.shape[0])
    try:
        nnet.fit(X=X_train, y=y_train)
    except KeyboardInterrupt:
        print("interruption...")
    light.endings()  # save the duration
    light.store_experiment()  # update the DB
    light.close()
