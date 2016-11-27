## File descriptions

## RN32_Paper_Ep200_Batch300_WeDe.txt
32 Layer Resnet, doesn't have the --augment flag in the header, uses the normal CIFAR10 dataset with weight decay, run for 200 epochs, batchsize 300.

### RN32_Aug_CIFARAug_Ep150.txt
32 Layer Resnet, has the --augment flag in the header, uses the 3x CIFAR10 Augmented Dataset, run for 150 epochs.

### RN32_CIFAR10CropFlip_ConstantW.txt
32 Layer Resnet, has the --augment flag in the header, uses random flipping and cropping of the CIFAR10 dataset, but with constant weight initialization.

### RN32_CIFAR10CropFlip_HeNormal.txt
32 Layer Resnet, has the --augment flag in the header, uses random flipping and cropping of the CIFAR10 dataset, with He initialization.

