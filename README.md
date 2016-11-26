## IE598-project

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
