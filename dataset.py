import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import tensorflow as tf
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

# def train_loader():
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1,pin_memory=True)

#     return trainloader

# def test_loader():
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1,pin_memory=True)

#     return testloader

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

class MNIST(data.Dataset):
    """MNIST dataset."""

    def __init__(self, X, y, transform=None):
    
        self.transform = transform
        self.X = torch.stack([self.transform(x) for x in X])
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx],self.y[idx])

    
def create_trainset(dataset_name, batch_size):
    if dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1,pin_memory=True)

        return trainloader

    if dataset_name.lower() == 'mnist':
        #TODO fix redundancy of x_test, y_test in case that we need only trainset in create_trainset() and need testset only in create_testset()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        x_train_rescaled = x_train/255
        x_train_mean = np.mean(x_train_rescaled)
        x_train_std = np.std(x_train_rescaled)

        y_train = y_train.astype(np.long) #since we create dataset downloaded from tensorflow not pytorch, this cast change type from uint8 to long whuch is compatible with loss function

        transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize(x_train_mean, x_train_std),transforms.Pad(padding=2,padding_mode='edge')]) 
        #must be in order, eg. ToTensor() come first because the following transforms require tensor type data.
        trainset = MNIST(x_train,y_train,transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1,pin_memory=True)
        return trainloader

def create_testset(dataset_name):
    if dataset_name.lower() == 'cifar10':
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1,pin_memory=True)

        return testloader

    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        x_test_rescaled = x_test/255
        x_test_mean = np.mean(x_test_rescaled)
        x_test_std = np.std(x_test_rescaled)

        y_test = y_test.astype(np.long)
        
        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(x_test_mean, x_test_std),transforms.Pad(padding=2,padding_mode='edge')])
        testset = MNIST(x_test,y_test,transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1,pin_memory=True)
        return testloader