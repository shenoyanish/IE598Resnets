#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy import linalg
from six.moves import cPickle as pickle
from skimage.io import imsave
from transform import Transform

import argparse
import glob
import numpy as np
import os
import sys
import h5py


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


def load_dataset(datadir='data'):
    train_data = np.load('%s/train_data.npy' % datadir)
    train_labels = np.load('%s/train_labels.npy' % datadir)
    test_data = np.load('%s/test_data.npy' % datadir)
    test_labels = np.load('%s/test_labels.npy' % datadir)

    return train_data, train_labels, test_data, test_labels

def load_hdf5():
    CIFAR10_data = h5py.File('CIFAR10.hdf5','r')
    train_data = np.float32(CIFAR10_data['X_train'][:])
    train_labels = np.int32(np.array(CIFAR10_data['Y_train'][:]))
    test_data = np.float32(CIFAR10_data['X_test'][:])
    test_labels = np.int32(np.array(CIFAR10_data['Y_test'][:]))
    return train_data, train_labels, test_data, test_labels

def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data')
    parser.add_argument('--whitening', action='store_true', default=False)
    parser.add_argument('--norm', type=int, default=1)
    args = parser.parse_args()
    print(args)

    # trans = Transform(args)

    # if not os.path.exists(args.outdir):
    #     os.mkdir(args.outdir)

    # prepare training dataset
    data = np.zeros((50000, 3 * 32 * 32), dtype=np.float)
    labels = []

    for i, data_fn in enumerate(
            sorted(glob.glob('cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels.extend(batch['labels'])

    #normalize data
    data = data/255.0

    data = data.reshape((50000, 3, 32, 32))
    labels = np.asarray(labels, dtype=np.int32)
    
    # np.save('%s/train_data' % args.outdir, np.asarray(data, dtype=np.float32))
    # np.save('%s/train_labels' % args.outdir, np.asarray(labels, dtype=np.int32))

    train_data = data
    train_label = labels

    test = unpickle('cifar-10-batches-py/test_batch')
    data = np.asarray(test['data'], dtype=np.float)
    #normalize data
    data = data/255.0

    data = data.reshape((10000, 3, 32, 32))
    labels = np.asarray(test['labels'], dtype=np.int32)
    # np.save('%s/test_data' % args.outdir, data)
    # np.save('%s/test_labels' % args.outdir, labels)

    test_data = data
    test_label = labels
    
    f = h5py.File('CIFAR10.hdf5','w')
    f.create_dataset('X_train',data=train_data,compression="gzip")
    f.create_dataset('Y_train',data=train_label,compression="gzip")
    f.create_dataset('X_test',data=test_data,compression="gzip")
    f.create_dataset('Y_test',data=test_label,compression="gzip")
    f.close()

    ## data written as np arrays
    # data = (nexamples,3,32,32), labels = (nexamples)




    # saving training dataset
    # if not os.path.exists('data/test_data'):
    #     os.mkdir('data/test_data')
    # for i in range(50000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     d = (d * 255).astype(np.uint8)
    #     imsave('data/test_data/train_{}.png'.format(i), d)



    # for i in range(10000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     d = (d * 255).astype(np.uint8)
    #     imsave('data/test_data/test_{}.png'.format(i), d)
