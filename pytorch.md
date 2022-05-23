# Pytorch Quick Snippets


Libraries

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

Get Datasets in torchvision

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
