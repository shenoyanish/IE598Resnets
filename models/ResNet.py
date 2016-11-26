#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import serializers
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import math
import os
import numpy as np

class Module(chainer.Chain):

    def __init__(self, n_in, n_out, stride=1):
        
        self.dtype = np.float32
        w = 1/np.sqrt(2)
        initW = initializers.HeNormal(scale=w)
        initbias = initializers.Zero()
        super(Module, self).__init__(
            conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, 1, initialW=initW, initial_bias=initbias),
            bn1=L.BatchNormalization(n_out,dtype=self.dtype),
            conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, 1, initialW=initW, initial_bias=initbias),
            bn2=L.BatchNormalization(n_out,dtype=self.dtype),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
            if x.data.shape[1] != h.data.shape[1]:
                x = F.concat((x, x * 0))
        return F.relu(h + x)

class Block(chainer.Chain):

    def __init__(self, n_in, n_out, n, stride=1):
        super(Block, self).__init__()
        links = [('m0', Module(n_in, n_out, stride))]
        links += [('m{}'.format(i + 1), Module(n_out, n_out))
                  for i in range(n - 1)]
        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            x = getattr(self, name)(x, train)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n=5):
        super(ResNet, self).__init__()
        self.dtype = np.float32
        w = 1/np.sqrt(2)
        initW = initializers.HeNormal(scale=w)
        initbias = initializers.Zero()
        
        links = [('conv1', L.Convolution2D(3, 16, 3, 1, 1, initialW=initW, initial_bias=initbias)),
                 ('bn2', L.BatchNormalization(16,dtype=self.dtype)),
                 ('_relu3', F.ReLU()),
                 ('res4', Block(16, 16, n)),
                 ('res5', Block(16, 32, n, 2)),
                 ('res6', Block(32, 64, n, 2)),
                 ('_apool7', F.AveragePooling2D(8, 1, 0, False, True)),
                 ('fc8', L.Linear(64, 10,initialW=initW, initial_bias=initbias))]
        for i,link in enumerate(links):
            if 'res' in link[0] and os.path.isfile(link[0]+'.hdf5'):
                self.add_link(*link)
                serializers.load_hdf5(link[0]+'.hdf5',getattr(self,link[0]))
            elif not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True

    def save(self):
        for name, f in self.forward:
            if 'res' in name:
                serializers.save_hdf5(name+'.hdf5',getattr(self,name))

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        for name, f in self.forward:
            # print name
            if 'res' in name:
                x = getattr(self, name)(x, self.train)
            elif name.startswith('bn'):
                x = getattr(self, name)(x, not self.train)
            elif name.startswith('_'):
                x = f(x)
            else:
                x = getattr(self, name)(x)
            # print x.data.shape
        if self.train:
            self.loss = F.softmax_cross_entropy(x, t)
            self.accuracy = F.accuracy(x, t)
            return self.loss
        else:
            return x

model = ResNet()
