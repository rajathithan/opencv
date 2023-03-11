# Pytorch Quick Snippets


### Libraries

```

import torch.nn as nn
import torch.optim as optim
import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Get reproducible results
torch.manual_seed(0)
```

### Get Datasets in torchvision

```
[print(x) for x in inspect.getmembers(torchvision.datasets,predicate=inspect.isclass)]

('CIFAR10', torchvision.datasets.cifar.CIFAR10)
('CIFAR100', torchvision.datasets.cifar.CIFAR100)
('Caltech101', torchvision.datasets.caltech.Caltech101)
('Caltech256', torchvision.datasets.caltech.Caltech256)
('CelebA', torchvision.datasets.celeba.CelebA)
('Cityscapes', torchvision.datasets.cityscapes.Cityscapes)
('CocoCaptions', torchvision.datasets.coco.CocoCaptions)
('CocoDetection', torchvision.datasets.coco.CocoDetection)
('DatasetFolder', torchvision.datasets.folder.DatasetFolder)
('EMNIST', torchvision.datasets.mnist.EMNIST)
('FakeData', torchvision.datasets.fakedata.FakeData)
('FashionMNIST', torchvision.datasets.mnist.FashionMNIST)
('Flickr30k', torchvision.datasets.flickr.Flickr30k)
('Flickr8k', torchvision.datasets.flickr.Flickr8k)
('HMDB51', torchvision.datasets.hmdb51.HMDB51)
('INaturalist', torchvision.datasets.inaturalist.INaturalist)
('ImageFolder', torchvision.datasets.folder.ImageFolder)
('ImageNet', torchvision.datasets.imagenet.ImageNet)
('KMNIST', torchvision.datasets.mnist.KMNIST)
('Kinetics', torchvision.datasets.kinetics.Kinetics)
('Kinetics400', torchvision.datasets.kinetics.Kinetics400)
('Kitti', torchvision.datasets.kitti.Kitti)
('LFWPairs', torchvision.datasets.lfw.LFWPairs)
('LFWPeople', torchvision.datasets.lfw.LFWPeople)
('LSUN', torchvision.datasets.lsun.LSUN)
('LSUNClass', torchvision.datasets.lsun.LSUNClass)
('MNIST', torchvision.datasets.mnist.MNIST)
('Omniglot', torchvision.datasets.omniglot.Omniglot)
('PhotoTour', torchvision.datasets.phototour.PhotoTour)
('Places365', torchvision.datasets.places365.Places365)
('QMNIST', torchvision.datasets.mnist.QMNIST)
('SBDataset', torchvision.datasets.sbd.SBDataset)
('SBU', torchvision.datasets.sbu.SBU)
('SEMEION', torchvision.datasets.semeion.SEMEION)
('STL10', torchvision.datasets.stl10.STL10)
('SVHN', torchvision.datasets.svhn.SVHN)
('UCF101', torchvision.datasets.ucf101.UCF101)
('USPS', torchvision.datasets.usps.USPS)
('VOCDetection', torchvision.datasets.voc.VOCDetection)
('VOCSegmentation', torchvision.datasets.voc.VOCSegmentation)
('VisionDataset', torchvision.datasets.vision.VisionDataset)
('WIDERFace', torchvision.datasets.widerface.WIDERFace)

```

### Loading a test and validation dataset

```
# Training set
train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

# Validation dataset
validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

# Batch size : How many images are used to calculate the gradient
batch_size = 32

# Train DataLoader 
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)
# Validation DataLoader 
validation_loader = DataLoader(dataset=validation_dataset, 
                               batch_size=batch_size, 
                               shuffle=False)

```


### Default Imports
```
import os
import time

from typing import Iterable, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
```

### Le-Net architecture
```
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self._body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Fully connected layers
        self._head = nn.Sequential(
            
            nn.Linear(in_features=16 * 5 * 5, out_features=120), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=120, out_features=84), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        # apply feature extractor
        x = self._body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weights_in_last conv_layer
        x = x.view(x.size()[0], -1)
        # apply classification head
        x = self._head(x)
        return x

```

### Display a network
```
lenet_model = LeNet()
print(lenet_model)

```

### Get the dataset and transform to tensors
```
def get_data(batch_size, data_root='data', num_workers=1):
    
    train_test_transforms = transforms.Compose([
        # Resize to 32X32
        transforms.Resize((32, 32)),
        # this re-scales image tensor values between 0-1. image_tensor /= 255
        transforms.ToTensor(),
        # subtract mean (0.2860) and divide by variance (0.3530).
        # This mean and variance is calculated on training data (verify for yourself)
        transforms.Normalize((0.2860, ), (0.3530, ))
    ])
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=data_root, train=False, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader
```
