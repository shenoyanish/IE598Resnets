#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from multiprocessing import Process
from multiprocessing import Queue

import argparse
import chainer
import draw_loss
import imp
import logging
import numpy as np
import os
import shutil
import six
import sys
import time

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transform import Transform
    from dataset import load_dataset,load_hdf5
else:
    import matplotlib.pyplot as plt
    from transform import Transform
    from dataset import load_dataset,load_hdf5


def create_result_dir(args):
    result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
    result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
    result_dir += str(time.time()).replace('.', '')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    args.log_fn = log_fn
    args.result_dir = result_dir


def get_model_optimizer(args):
    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model

    dst = '%s/%s' % (args.result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(args.model, dst)

    dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # prepare optimizer
    if 'opt' in args:
        # prepare optimizer
        if args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
        elif args.opt == 'Adam':
            optimizer = optimizers.Adam(alpha=args.alpha)
        elif args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
        else:
            raise Exception('No optimizer is selected')

        optimizer.setup(model)

        if args.opt == 'MomentumSGD':
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(args.weight_decay))
        return model, optimizer
    else:
        print('No optimizer generated.')
        return model


def augmentation(args, aug_queue, data, label, train):
    trans = Transform(args)
    np.random.seed(int(time.time()))
    perm = np.random.permutation(data.shape[0])
    if train:
        for i in six.moves.range(0, data.shape[0], args.batchsize):
            chosen_ids = perm[i:i + args.batchsize]
            x = np.asarray(data[chosen_ids], dtype=np.float32)
            x = x.transpose((0, 3, 1, 2))
            t = np.asarray(label[chosen_ids], dtype=np.int32)
            aug_queue.put((x, t))
    else:
        for i in six.moves.range(data.shape[0]):
            aug = trans(data[i])
            x = np.asarray(aug, dtype=np.float32).transpose((0, 3, 1, 2))
            t = np.asarray(np.repeat(label[i], len(aug)), dtype=np.int32)
            aug_queue.put((x, t))
    aug_queue.put(None)
    return

def random_crop_flip(xbatch):
    padding  = 4
    padded_images = np.zeros((xbatch.shape[0],xbatch.shape[1],xbatch.shape[2]+2*padding,xbatch.shape[3]+2*padding),dtype=np.float32)
    cropped_images = np.zeros((xbatch.shape[0],xbatch.shape[1],xbatch.shape[2],xbatch.shape[3]),dtype=np.float32)
    crop_x0 = np.random.randint(0,2*padding+1,size=(xbatch.shape[0],1),dtype=np.int32)
    crop_y0 = np.random.randint(0,2*padding+1,size=(xbatch.shape[0],1),dtype=np.int32)
    flip_p = np.random.random(size=(xbatch.shape[0],1))
    padded_images[:,:,padding:padding+xbatch.shape[2],padding:padding+xbatch.shape[3]] = xbatch[:,:,:,:]
    for j in range(xbatch.shape[0]):
        cropped_images[j,:,:,:] =  padded_images[j,:,int(crop_x0[j]):int(crop_x0[j])+xbatch.shape[2],int(crop_y0[j]):int(crop_y0[j])+xbatch.shape[3]]
        if flip_p[j]>=0.5:
            cropped_images[j,:,:,:] = cropped_images[j,:,:,::-1]
    assert(cropped_images.shape == xbatch.shape)
    
    return cropped_images

def one_epoch_resnet(args,model,optimizer,data,label,epoch,train):
    model.train = train

    xp = cuda.cupy if args.gpu >= 0 else np

    
    GPU_on = True if args.gpu >= 0 else False
    # logging.info('data loading started')

    #transfered model to gpu in get_model_optimizer
    #optimizer setup model called in get_model_optimizer

    L_Y_train = len(label)
    time1 = time.time()

    np.random.seed(int(time.time()))
    perm = np.random.permutation(data.shape[0])
    data = data[perm,:]
    label = label[perm]

    sum_accuracy = 0
    sum_loss = 0
    num = 0
    for i in range(0,data.shape[0],args.batchsize):
        data_batch = data[i:i+args.batchsize,:]
        label_batch =label[i:i+args.batchsize]

        if train:
            data_batch = random_crop_flip(data_batch)

        if (GPU_on):
            data_batch = cuda.to_gpu(data_batch)
            label_batch = cuda.to_gpu(label_batch)
        # else:
        #     data_batch = data[i:i+args.batchsize,:]
        #     label_batch =label[i:i+args.batchsize]

        # convert to chainer variable
        x = Variable(xp.asarray(data_batch))
        t = Variable(xp.asarray(label_batch))

        model.zerograds()
        
        # total_accuracy += 100.0*np.float(accuracy.data)
        # batch_counter += 1
        # num +=data_batch.shape[0]

        if train:

            #loss= model(x,t)
            #loss.backward()
            #optimizer.update()

            optimizer.update(model, x, t)

            if epoch == 1 and num == 0:
                with open('{}/graph.dot'.format(args.result_dir), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (model.loss, ), remove_split=True)
                    o.write(g.dump())
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
            num += t.data.shape[0]
            logging.info('{:05d}/{:05d}\tloss:{:.3f}\tacc:{:.3f}'.format(
                num, data.shape[0], sum_loss / num, sum_accuracy / num))
        else:
            pred = model(x, t).data
            sum_accuracy +=  float(sum(pred.argmax(axis=1) == t.data))
            num += t.data.shape[0]
        del x,t

    if train and (epoch == 1 or epoch % args.snapshot == 0):
        model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
        opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
        serializers.save_hdf5(model_fn, model)
        serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info('epoch:{}\ttrain loss:{:.4f}\ttrain accuracy:{:.4f}'.format(
            epoch, sum_loss / num, sum_accuracy / num))
    else:
        logging.info('epoch:%d\ttest accuracy:%0.4f'%(epoch,sum_accuracy/num))

def one_epoch(args, model, optimizer, data, label, epoch, train):
    model.train = train
    xp = cuda.cupy if args.gpu >= 0 else np

    # # for parallel augmentation
    # aug_queue = Queue()
    # aug_worker = Process(target=augmentation,
    #                      args=(args, aug_queue, data, label, train))
    # aug_worker.start()
    
    logging.info('data loading started')

    sum_accuracy = 0
    sum_loss = 0
    num = 0
    while True:
        datum = aug_queue.get()
        if datum is None:
            break
        x, t = datum

        volatile = 'off' if train else 'on'
        x = Variable(xp.asarray(x), volatile=volatile)
        t = Variable(xp.asarray(t), volatile=volatile)

        if train:
            optimizer.update(model, x, t)
            if epoch == 1 and num == 0:
                with open('{}/graph.dot'.format(args.result_dir), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (model.loss, ), remove_split=True)
                    o.write(g.dump())
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
            num += t.data.shape[0]
            logging.info('{:05d}/{:05d}\tloss:{:.3f}\tacc:{:.3f}'.format(
                num, data.shape[0], sum_loss / num, sum_accuracy / num))
        else:
            pred = model(x, t).data
            pred = pred.mean(axis=0)
            acc = int(pred.argmax() == t.data[0])
            sum_accuracy += acc
            num += 1
            logging.info('{:05d}/{:05d}\tacc:{:.3f}'.format(
                num, data.shape[0], sum_accuracy / num))

        del x, t

    if train and (epoch == 1 or epoch % args.snapshot == 0):
        model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
        opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
        serializers.save_hdf5(model_fn, model)
        serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info('epoch:{}\ttrain loss:{}\ttrain accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))
    else:
        logging.info('epoch:{}\ttest loss:{}\ttest accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))

    draw_loss.draw_loss_curve('{}/log.txt'.format(args.result_dir),
                              '{}/log.png'.format(args.result_dir), epoch)

    aug_worker.join()
    logging.info('data loading finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/VGG.py')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=170)
    parser.add_argument('--batchsize', type=int, default=128) #128
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--augment', action='store_true', default=False)

    # optimization
    parser.add_argument('--opt', type=str, default='MomentumSGD',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay_freq', type=int, default=80)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1701)

    args = parser.parse_args()
    np.random.seed(args.seed)
    # os.environ['CHAINER_TYPE_CHECK'] = str(args.type_check)
    # os.environ['CHAINER_SEED'] = str(args.seed)

    # create result dir
    create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)

    dataset = load_hdf5(args.augment)
    tr_data, tr_labels, te_data, te_labels = dataset

    # learning loop
    for epoch in range(1, args.epoch + 1):
        logging.info('learning rate:{}'.format(optimizer.lr))

        one_epoch_resnet(args,model,optimizer,tr_data,tr_labels,epoch,True)

        # one_epoch(args, model, optimizer, tr_data, tr_labels, epoch, True)

        if epoch == 1 or epoch % args.validate_freq == 0:
            one_epoch_resnet(args, model, optimizer, te_data, te_labels, epoch, False)

        if args.opt == 'MomentumSGD' and epoch % args.lr_decay_freq == 0:
            optimizer.lr *= args.lr_decay_ratio
