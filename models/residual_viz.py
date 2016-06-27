#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)

With n=5, i.e. 32-layer network from the paper, this achieves a validation error of 6.88% (vs 7.51% in the paper).
The accuracy has not yet been tested for the other values of n.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    # shuffle arrays
    from random import shuffle
    train_index = [i for i in range(100000)]
    test_index = [i for i in range(10000)]
    random.shuffle(train_index)
    random.shuffle(test_index)
    train_index = np.array(train_index)
    test_index = np.array(test_index)
    
    X_train = X_train[train_index,:,:,:]
    Y_train = Y_train[train_index]
    X_test = x[test_index+50000,:,:,:]
    Y_test = y[test_index+50000]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

# ##################### Build the neural network model #######################

#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify

# NB! from pull request #461 : https://github.com/f0k/Lasagne/blob/98b5581fa830cda3d3f838506ef14e5811a35ef7/lasagne/layers/normalization.py
from normalization import batch_norm

def build_cnn(input_var=None, n=5):
    
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
                block = NonlinearityLayer(batch_norm(ElemwiseSumLayer([stack_2, projection])),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                # we use a pooling layer to get identity with strides, since identity layers with stride don't exist in Lasagne
                identity = Pool2DLayer(l, pool_size=1, stride=(2,2), mode='average_exc_pad')
                padding = PadLayer(identity, [out_num_filters/4,0,0], batch_ndim=1)
                block = NonlinearityLayer(batch_norm(ElemwiseSumLayer([stack_2, padding])),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(batch_norm(ElemwiseSumLayer([stack_2, l])),nonlinearity=rectify)
        
        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            nonlinearity=softmax)

    return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################

def main(n=5, num_epochs=82):
    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network))
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss = loss + l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=sh_lr, momentum=0.9)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, 128, shuffle=True, augment=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # adjust learning rate as in paper
        # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
        if (epoch+1) == 41 or (epoch+1) == 61:
            new_lr = sh_lr.get_value() * 0.1
            print("New LR:"+str(new_lr))
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # dump the network weights to a file :
    np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('cifar10_deep_residual_model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    from lasagnekit.misc.draw_net import draw_to_file
    from lasagne.layers import get_all_layers
    import residual
    import residualv2
    import residualv3
    import residualv4

    from hp_toolkit.hp import instantiate_default

    cnn = build_cnn(input_var=None, n=5)
    layers = get_all_layers(cnn)
    draw_to_file(layers, "residual_other.svg")

    hp = instantiate_default(residual.params)
    cnn = residual.build_model(**hp)
    cnn = cnn.output_layers[0]
    layers = get_all_layers(cnn)
    draw_to_file(layers, "residual.svg")

    hp = instantiate_default(residualv2.params)
    cnn = residualv2.build_model(**hp)
    cnn = cnn.output_layers[0]
    layers = get_all_layers(cnn)
    draw_to_file(layers, "residualv2.svg")

    hp = instantiate_default(residualv3.params)
    cnn = residualv3.build_model(**hp)
    cnn = cnn.output_layers[0]
    layers = get_all_layers(cnn)
    draw_to_file(layers, "residualv3.svg")

    hp = instantiate_default(residualv4.params)
    cnn = residualv4.build_model(**hp)
    cnn = cnn.output_layers[0]
    layers = get_all_layers(cnn)
    draw_to_file(layers, "residualv4.svg")
