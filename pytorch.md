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

### Get the dataset and transform to tensors & normalize the values
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

### System Configuration
```
@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed: int = 21  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)
    
```

### Training Configuration
```
@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 64  # amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 30  # number of times the whole dataset will be passed through the network
    learning_rate: float = 0.001  # determines the speed of network's weights update
    log_interval: int = 500  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./data"  # folder to save Fashion MNIST data (default: data)
    num_workers: int = 10  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
    
    
```


### System configuration
```
def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
```

### Train the model
```
def train(
    train_config: TrainingConfiguration, model: nn.Module, optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader, epoch_idx: int
) -> Tuple[float, float]:
    
    # change model in training mode
    model.train()
    
    # to get batch loss
    batch_loss = np.array([])
    
    # to get batch accuracy
    batch_acc = np.array([])
        
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # clone target
        indx_target = target.clone()
        # send data to device (it is mandatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)

        # reset parameters gradient to zero
        optimizer.zero_grad()
        
        # forward pass to the model
        output = model(data)
        
        # cross entropy loss
        loss = F.cross_entropy(output, target)
        
        # find gradients w.r.t training parameters
        loss.backward()
        
        # Update parameters using gradients
        optimizer.step()
        
        batch_loss = np.append(batch_loss, [loss.item()])
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
            
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]  
                        
        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()
            
        # accuracy
        acc = float(correct) / float(len(data))
        
        batch_acc = np.append(batch_acc, [acc])

#         if batch_idx % train_config.log_interval == 0 and batch_idx > 0:              
#             print(
#                 'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
#                     epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
#                 )
#             )
            
    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    
    print('\nEpoch: {} Loss: {:.6f} Acc: {:.4f}'.format(epoch_idx, epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc
```

### Validate the model
```
def validate(
    train_config: TrainingConfiguration,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    # 
    model.eval()
    test_loss = 0
    count_corect_predictions = 0
    for data, target in test_loader:
        indx_target = target.clone()
        data = data.to(train_config.device)
        
        target = target.to(train_config.device)
        
        output = model(data)
        # add loss for each mini batch
        test_loss += F.cross_entropy(output, target).item()
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
        
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1] 
        
        # add correct prediction count
        count_corect_predictions += pred.cpu().eq(indx_target).sum()

    # average over number of mini-batches
    test_loss = test_loss / len(test_loader)  
    
    # average over number of dataset
    accuracy = 100. * count_corect_predictions / len(test_loader.dataset)
    
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, count_corect_predictions, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy/100.0

```

### Main Function
```
def main(model, optimizer, system_configuration=SystemConfiguration(), 
         training_configuration=TrainingConfiguration()):
    
    # system configuration
    setup_system(system_configuration)

    # batch size
    batch_size_to_set = training_configuration.batch_size
    # num_workers
    num_workers_to_set = training_configuration.num_workers
    # epochs
    epoch_num_to_set = training_configuration.epochs_count

    # if GPU is available use training config, 
    # else lower batch_size, num_workers and epochs count
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2

    # data loader
    train_loader, test_loader = get_data(
        batch_size=batch_size_to_set,
        data_root=training_configuration.data_root,
        num_workers=num_workers_to_set
    )
    
    # Update training configuration
    training_configuration = TrainingConfiguration(
        device=device,
        batch_size=batch_size_to_set,
        num_workers=num_workers_to_set
    )
        
    # send model to device (GPU/CPU)
    model.to(training_configuration.device)

    best_loss = torch.tensor(np.inf)
    
    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])
    
    # epoch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])
    
    # trainig time measurement
    t_begin = time.time()
    for epoch in range(training_configuration.epochs_count):
        
        train_loss, train_acc = train(training_configuration, model, optimizer, train_loader, epoch)
        
        epoch_train_loss = np.append(epoch_train_loss, [train_loss])
        
        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs_count - elapsed_time
        
        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )
    
    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc
```

### SGD
```
For stochastic gradient descent update, we use the following method in PyTorch:

torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups. model.parameters() gives iterable model parameters. [Required]

lr (python:float) – learning rate [Required]

momentum (python:float, optional) – momentum factor (default: 0)

weight_decay (python:float, optional) – weight decay (L2 penalty) (default: 0)

dampening (python:float, optional) – dampening for momentum (default: 0)

nesterov (bool, optional) – enables Nesterov momentum (default: False)

======================================================================

model = LeNet()

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss_sgd, train_acc_sgd, test_loss_sgd, test_acc_sgd = main(model, optimizer)
```

### SGD with momentum
```
In PyTorch, we use torch.optim.SGD with non-zero momentum value.

In the following training, we will use 
β=0.9. Here β is momentum.


================================================================================
model = LeNet()

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate,
    # to use momentum, changed default value 0 to 0.9
    momentum = 0.9
)


model, train_loss_sgd_momentum, train_acc_sgd_momentum, test_loss_sgd_momentum, test_acc_sgd_momentum = main(model,optimizer)
```
### RMSProp
```
For RMSProp weight update, we use the following method in PyTorch:

torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups

lr (python:float, optional) – learning rate (default: 1e-2). 

α is learning rate in the RMSProp update rule.

momentum (python:float, optional) – momentum factor (default: 0)

alpha (python:float, optional) – smoothing constant (default: 0.99). 

β is smothing constant in the RMSProp update rule.

eps (python:float, optional) – term added to the denominator to improve numerical stability (default: 1e-8). 

This ϵ is the RMSProp update rule.

centered (bool, optional) – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

weight_decay (python:float, optional) – weight decay (L2 penalty) (default: 0)

=====================================================================================================

model = LeNet()

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.RMSprop(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss_rms_prop, train_acc_rms_prop, test_loss_rms_prop, test_acc_rms_prop = main(model, optimizer)

```

### ADAM = Momentum + RMSPROP
```
For Adam weight update, we use the following method in PyTorch:

torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups

lr (python:float, optional) – learning rate (default: 1e-3). 
α is learning rate in Adam update rule.

betas (Tuple[python:float, python:float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)). In Adam update rule, first value of tuple is β1  and second value is β2

weight_decay (python:float, optional) – weight decay (L2 penalty) (default: 0)

amsgrad (boolean, optional) – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)

========================================================

model = LeNet()

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss_adam, train_acc_adam, test_loss_adam, test_acc_adam = main(model, optimizer)

```

