## IE 598 Project, Neural Networks and Deep Learning
Implementation of Residual Neural Networks using Chainer. This code reproduces results from Deep Residual Learning for Image Recognition by He et al. on the CIFAR-10 dataset.

## Download the data and save it as HDF5
```
bash download.sh
```
This generates CIFAR10.hdf5 and its augmented version CIFAR10_augmented.hdf5. 

## Training the resnet
```
python train.py --model models/ResNet.py --gpu 0
```

To use the augmented dataset use the flag --augment, for example,
```
python train.py --augment --model models/ResNet.py --gpu 0
```
