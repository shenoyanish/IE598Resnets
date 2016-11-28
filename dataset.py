#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
from __future__ import unicode_literals
from scipy import linalg
from six.moves import cPickle as pickle
from skimage.io import imsave
from transform import Transform
from scipy import ndimage

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

def load_hdf5(is_dataaugment):

    CIFAR10_data = h5py.File('CIFAR10.hdf5','r')
    if is_dataaugment:
        CIFAR10_data = h5py.File('CIFAR10_augmented.hdf5','r')
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

def meansubtract(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    return mdata


def augment_data(image):
    temp_image = np.rollaxis(image, 0, 3)

    # augmentation
    temp_image_flr = np.fliplr(temp_image)  # flip
    temp_image_con = (temp_image - temp_image.min()) / (temp_image.max() - temp_image.min())  # normalize
    temp_image_med = ndimage.median_filter(temp_image, 2)  # median filter

    # plot
    # plt.interactive(False)
    # plt.figure(1)
    # plt.subplot(141)
    # plt.imshow(temp_image)
    # plt.subplot(142)
    # plt.imshow(temp_image_flr)
    # plt.subplot(143)
    # plt.imshow(temp_image_con)
    # plt.subplot(144)
    # plt.imshow(temp_image_med)
    # plt.show()


    # restore
    temp_image_flr = np.rollaxis(temp_image_flr, 2, 0)
    temp_image_con = np.rollaxis(temp_image_con, 2, 0)
    temp_image_med = np.rollaxis(temp_image_med, 2, 0)

    return temp_image_flr,temp_image_con,temp_image_med



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
    #data = data/255.0

#     if args.whitening:
#         components, mean, data = preprocessing(data)
#         np.save('{}/components'.format(args.outdir), components)
#         np.save('{}/mean'.format(args.outdir), mean)
        
    data = meansubtract(data)

    data = data.reshape((50000, 3, 32, 32))
    labels = np.asarray(labels, dtype=np.int32)
    
    train_data = data
    train_label = labels

    # read the test dataset
    test = unpickle('cifar-10-batches-py/test_batch')
    data = np.asarray(test['data'], dtype=np.float)
    
    #normalize data
    #data = data/255.0


#     if args.whitening:
#         components, mean, data = preprocessing(data)
#         np.save('{}/components'.format(args.outdir), components)
#         np.save('{}/mean'.format(args.outdir), mean)
    
    data = meansubtract(data)
    data = data.reshape((10000, 3, 32, 32))
    labels = np.asarray(test['labels'], dtype=np.int32)

    test_data = data
    test_label = labels
    
    f = h5py.File('CIFAR10.hdf5','w')
    f.create_dataset('X_train',data=train_data,compression="gzip")
    f.create_dataset('Y_train',data=train_label,compression="gzip")
    f.create_dataset('X_test',data=test_data,compression="gzip")
    f.create_dataset('Y_test',data=test_label,compression="gzip")
    f.close()

    ## data written as np arrays
    ## data = (nexamples,3,32,32), labels = (nexamples)

    # augmentation
    X_train = train_data
    Y_train = train_label
    X_test = test_data
    Y_test = test_label

    unique,counts = np.unique(Y_train,return_counts=True)
    original_counts = dict(zip(unique,counts))

    # augment the data
    new_x_train = X_train
    new_y_train = Y_train
    new_x_train = list(new_x_train)
    new_y_train = list(new_y_train)

    # iterate through all training examples
    for i in range(X_train.shape[0]):
        print i
        temp_image = X_train[i]
        temp_class = Y_train[i]
        temp_image_flr,temp_image_con,temp_image_med = augment_data(temp_image)     
        new_x_train.append(temp_image_flr)
        new_x_train.append(temp_image_con)
        new_x_train.append(temp_image_med)
        new_y_train.append(temp_class)
        new_y_train.append(temp_class)
        new_y_train.append(temp_class)
        # np.concatenate((new_x_train,(temp_image_flr,temp_image_med,temp_image_con)),axis=0)
        # np.concatenate((new_y_train,(temp_class,temp_class,temp_class)),axis=0)

    X_train = np.array(new_x_train,dtype=np.float32)
    Y_train = np.array(new_y_train,dtype=np.int32)
    print X_train.shape
    print Y_train.shape

    unique,counts = np.unique(Y_train,return_counts=True)
    new_counts = dict(zip(unique,counts))

    print original_counts
    print new_counts

    for class_label in original_counts.keys():
        if 4*original_counts[class_label] != new_counts[class_label]:
            print 'Error, augmented dataset does not have required distribution'

    #save as HDF5 files
    g = h5py.File('CIFAR10_augmented.hdf5', 'w')  
    g.create_dataset('X_train', data = X_train, compression = "gzip")
    g.create_dataset('Y_train', data = Y_train, compression = "gzip")
    g.create_dataset('X_test', data = X_test, compression = "gzip")
    g.create_dataset('Y_test', data = Y_test, compression = "gzip")
    g.close()
