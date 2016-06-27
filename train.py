from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')   # NOQA
from lasagnekit.easy import BatchOptimizer, BatchIterator, get_batch_slice
from lasagnekit.nnet.capsule import Capsule
from lasagnekit.easy import iterate_minibatches
from lasagne import updates
from lasagnekit.updates import santa_sss
updates.santa_sss = santa_sss  # NOQA
import theano
import theano.tensor as T

import numpy as np
import json

from skimage.io import imsave
from lasagnekit.datasets.infinite_image_dataset import Transform


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
            tr, _ = Transform(X.transpose(0, 2, 3, 1),
                              np.random,
                              **self.transform_params)
            imsave("out.png", (((tr[0] + 1) / 2.)))
            X_transformed = tr.transpose((0, 3, 1, 2))
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
    from hp_toolkit.hp import (
         Param, make_constant_param,
         instantiate_random, instantiate_default
    )
    import argparse
    from models import vgg  # NOQA
    from models import vgg_small  # NOQA
    from models import vgg_very_small  # NOQA
    from models import spatially_sparse  # NOQA
    from models import nin  # NOQA
    from models import fully  # NOQA
    from models import residual  # NOQA
    from models import residualv2  # NOQA
    from models import residualv3  # NOQA
    from models import residualv4  # NOQA
    from models import residualv5  # NOQA
    from lightjob.cli import load_db

    db = load_db()
    job_content = {}

    parser = argparse.ArgumentParser(description='zoo')
    parser.add_argument("--budget-hours",
                        default=np.inf,
                        help="nb of maximum hours (defaut=inf)")
    parser.add_argument("--fast-test", default=False, type=bool)
    parser.add_argument("--model", default="vgg", type=str)
    parser.add_argument("--default-model", default=False, type=bool)

    models = {
        "vgg": vgg,
        "vgg_small": vgg_small,
        "vgg_very_small": vgg_very_small,
        "spatially_sparse": spatially_sparse,
        "nin": nin,
        "fully": fully,
        "residual": residual,
        "residualv2": residualv2,
        "residualv3": residualv3,
        "residualv4": residualv4,
        "residualv5": residualv5
    }
    args = parser.parse_args()
    model_class = models[args.model]
    budget_sec = args.budget_hours * 3600
    begin = datetime.now()
    seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    fast_test = args.fast_test
    rng = np.random

    if args.default_model is True:
        instantiate = instantiate_default
    else:
        instantiate = instantiate_random

    job = {}

    data = Cifar10(batch_indexes=[1, 2, 3, 4, 5])
    data.load()

    data_test = Cifar10(batch_indexes=[6])
    data_test.load()

    job['dataset'] = data.__class__.__name__

    hp = dict(
        learning_rate=Param(initial=0.001, interval=[-4, -2], type='real', scale='log10'),
        learning_rate_decay=Param(initial=0.05, interval=[0, 0.1], type='real'),
        learning_rate_decay_method=Param(initial='discrete', interval=['exp', 'none', 'sqrt', 'lin', 'discrete'], type='choice'),
        momentum=Param(initial=0.9, interval=[0.5, 0.99], type='real'),
        #weight_decay=Param(initial=0, interval=[-10, -3], type='real', scale='log10'),
        weight_decay=make_constant_param(0.),
        discrete_learning_rate_epsilon=make_constant_param(1e-4),#NEW TO ADD
        discrete_learning_divide=make_constant_param(10.),
        l2_decay=Param(initial=0, interval=[-8, -4], type='real', scale='log10'),#NEW TO ADD
        max_epochs=make_constant_param(1000),
        batch_size=Param(initial=32,
                         interval=[16, 32, 64, 128],
                         type='choice'),
        patience_nb_epochs=make_constant_param(50),
        valid_ratio=make_constant_param(0.15),

        patience_threshold=make_constant_param(1),
        patience_check_each=make_constant_param(1),

        optimization=Param(initial='adam',
                           interval=['adam', 'nesterov_momentum', 'rmsprop'],
                           type='choice'),
        # data augmentation
        nb_data_augmentation=Param(initial=1, interval=[0, 1, 2, 3, 4], type='choice'),
        zoom_range=make_constant_param((1, 1)),
        rotation_range=make_constant_param((0, 0)),
        shear_range=make_constant_param((1, 1)),
        translation_range=make_constant_param((-5, 5)),
        do_flip=make_constant_param(True)
    )

    if fast_test is True:
        instantiate = instantiate_default

    default_params = {}
    if fast_test is True:
        default_params["max_epochs"] = 1
    hp = instantiate(hp, default_params=default_params)
    job_content['hp'] = hp

    hp_model = model_class.params
    hp_model = instantiate(hp_model)
    job_content['hp_model'] = hp_model

    model = model_class.build_model(
        input_width=data.img_dim[1],
        input_height=data.img_dim[2],
        output_dim=data.output_dim,
        **hp_model)
    job_content['model'] = model_class.__name__
    print(model_class.__name__)
    print(json.dumps(hp, indent=4))
    print(json.dumps(hp_model, indent=4))

    initial_lr = hp["learning_rate"]

    def evaluate(X, y, batch_size=None):
        if batch_size is None:
            batch_size = hp["batch_size"]
        y_pred = []
        for mini_batch in iterate_minibatches(X.shape[0],
                                              batch_size):
            y_pred.extend((nnet.predict(X[mini_batch]) == y[mini_batch]).tolist())
        return np.mean(y_pred)

    class MyBatchOptimizer(BatchOptimizer):

        def quitter(self, update_status):
            quit = super(MyBatchOptimizer, self).quitter(update_status)
            if (datetime.now() - begin).total_seconds() >= budget_sec:
                print("Budget finished.quit.")
                quit = True
            return quit

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            start = datetime.now()
            status = super(MyBatchOptimizer, self).iter_update(epoch,
                                                               nb_batches,
                                                               iter_update_batch)
            duration = (datetime.now() - start).total_seconds()
            status["duration"] = duration
            acc = evaluate(X_train, y_train, batch_size=self.batch_size_eval)
            status["accuracy_train"] = acc
            status["accuracy_train_std"] = 0
            acc = evaluate(X_valid, y_valid, batch_size=self.batch_size_eval)
            status["accuracy_valid"] = acc
            status["accuracy_valid_std"] = 0

            status["error_valid"] = 1 - status["accuracy_valid"]

            status = self.add_moving_avg("accuracy_train", status)
            status = self.add_moving_var("accuracy_train", status)
            status = self.add_moving_avg("accuracy_valid", status)
            status = self.add_moving_var("accuracy_valid", status)

            for k, v in status.items():
                if k not in job_content:
                    job_content[k] = [v]
                else:
                    job_content[k].append(v)

            lr = self.learning_rate
            lr_decay_method = hp["learning_rate_decay_method"]
            lr_decay = hp["learning_rate_decay"]
            cur_lr = lr.get_value()
            t = status["epoch"]

            if lr_decay_method == "exp":
                new_lr = cur_lr * (1 - lr_decay)
            elif lr_decay_method == "lin":
                new_lr = initial_lr / (1 + t)
            elif lr_decay_method == "sqrt":
                new_lr = initial_lr / np.sqrt(1 + t)
            elif lr_decay_method == 'discrete':
                eps = hp["discrete_learning_rate_epsilon"]
                div = hp["discrete_learning_divide"]
                if status["moving_var_accuracy_valid"] <= eps:
                    new_lr = cur_lr / div
                else:
                    new_lr = cur_lr
            else:
                new_lr = cur_lr

            new_lr = np.array(new_lr, dtype="float32")
            lr.set_value(new_lr)
            if 'learning_rate_per_epoch' not in job_content:
                job_content['learning_rate_per_epoch'] = []
            job_content['learning_rate_per_epoch'].append(float(self.learning_rate.get_value()))
            return status

        def add_moving_avg(self, name, status, B=0.9):
            if len(self.stats) >= 2:
                old_avg = self.stats[-2]["moving_avg_" + name]
            else:
                old_avg = 0
            avg = B * old_avg + (1 - B) * status[name]
            status["moving_avg_" + name] = avg
            return status

        def add_moving_var(self, name, status, B=0.9):
            if len(self.stats) >= 2:
                old_avg = self.stats[-2]["moving_avg_" + name]
                old_var = self.stats[-2]["moving_var_" + name]
            else:
                old_avg = 0
                old_var = 0
            new_avg = B * old_avg + (1 - B) * status[name]
            var = B * old_var + (1 - B) * (status[name] - old_avg) * (status[name] - new_avg)
            status["moving_var_" + name] = var
            return status

    learning_rate = theano.shared(np.array(hp["learning_rate"],
                                  dtype="float32"))
    momentum = hp["momentum"]

    optim_params = {"learning_rate": learning_rate}
    if "momentum" in hp["optimization"]:
        optim_params["momentum"] = hp["momentum"]

    batch_optimizer = MyBatchOptimizer(
        verbose=1, max_nb_epochs=hp["max_epochs"],
        batch_size=hp["batch_size"],
        optimization_procedure=(getattr(updates, hp["optimization"]),
                                optim_params),
        patience_stat="error_valid",
        patience_nb_epochs=hp["patience_nb_epochs"],
        patience_progression_rate_threshold=hp["patience_threshold"],
        patience_check_each=hp["patience_check_each"],
        verbose_stat_show=[
            "epoch",
            "duration",
            "accuracy_train",
            "accuracy_train_std",
            "accuracy_valid",
            "accuracy_valid_std",
        ]
    )
    batch_size_eval = 1024
    job_content['batch_size_eval'] = batch_size_eval
    batch_optimizer.learning_rate = learning_rate
    batch_optimizer.batch_size_eval = batch_size_eval

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

        if hp["l2_decay"] > 0:
            l2 = sum(T.sqr(param).sum() for param in model.capsule.all_params_regularizable) * hp["l2_decay"]
        else:
            l2 = 0

        return T.nnet.categorical_crossentropy(y_hat, y).mean() + l1 + l2

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

    from sklearn.preprocessing import LabelEncoder

    imshape = ([data.X.shape[0]] +
               list(data.img_dim))
    X = data.X.reshape(imshape).astype(np.float32)
    y = data.y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.astype(np.int32)

    X, y = shuffle(X, y)

    if fast_test is True:
        X = X[0:100]
        y = y[0:100]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=hp["valid_ratio"])

    # rescaling to [-1, 1]
    X_min = X_train.min(axis=(0, 2, 3))[None, :, None, None]
    X_max = X_train.max(axis=(0, 2, 3))[None, :, None, None]
    def preprocess(a):
        return (a / 255.) * 2 - 1
        # return 2 * ((a - X_min) / (X_max - X_min)) - 1
    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid)

    job_content['nb_examples_train'] = X_train.shape[0]
    job_content['nb_examples_valid'] = X_valid.shape[0]
    try:
        nnet.fit(X=X_train, y=y_train)
    except KeyboardInterrupt:
        print("interruption...")

    imshape = ([data_test.X.shape[0]] +
               list(data_test.img_dim))
    X_test = data_test.X.reshape(imshape).astype(np.float32)
    X_test = preprocess(X_test)
    y_test = data_test.y
    y_test = label_encoder.transform(y_test)
    y_test = y_test.astype(np.int32)

    acc = evaluate(X_test, y_test, batch_size_eval)
    job_content['accuracy_test'] = acc
    job_content['accuracy_test_std'] = 0
    print("Test accuracy : {}+-{}".format(acc, 0))
    if fast_test is False:
        db.add_job(job_content)
