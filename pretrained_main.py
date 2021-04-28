import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import os
import argparse

from PIL import Image

import glob
import json

#import relative .py files
import dataset
import models
import tucker_decomposition

parser = argparse.ArgumentParser(description='PyTorch CIFAR10')

parser.add_argument('--model_path', type=str, help='path to model')
parser.add_argument('--model', type=str, help='model type eg. Lenet, VGG16')
parser.add_argument('--dataset', default=None, type=str,help='dataset choices = ["MNIST","CIFAR10"]')
parser.add_argument('--batch_size', default=128, type=int,help='batch size')
parser.add_argument('--fine_tune', action='store_true',help='do fine tuning')
parser.add_argument('--save', action='store_true',help='save decomp model')
parser.add_argument('--epoch', type=int, default=5, help='epoch for fine tuning')
parser.add_argument('--lr', type=float, default=0.001, help='epoch for fine tuning')

args = parser.parse_args()

def measure_time(model,repetition):
    #model = model.to(device)
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0
    total_time = 0
    time_per_dataset = []
    time_per_data = []
    #warm up gpu
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
    for i in range(repetition):
    #measure time
        total = 0
        total_time = 0
        with torch.no_grad():
          for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            starter.record()
            output = model(inputs)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            total_time += curr_time

            total += targets.size(0)

        time_per_dataset.append(total_time)
        time_per_data.append(total_time/total)

    model = model.cpu()
    print(time_per_dataset)
    print(time_per_data)
    final_time_per_dataset = np.mean(time_per_dataset)
    final_time_per_data = np.mean(time_per_data)
    std_time_per_dataset = np.std(time_per_dataset)
    std_time_per_data = np.std(time_per_data)

    return final_time_per_dataset, final_time_per_data, std_time_per_dataset, std_time_per_data

def test(model):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

      #print(f'model acuracy: {correct/total}')
    #model = model.to('cpu')
    return correct/total

def fine_tune(model,lr=0.001, max_iter=30):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),lr = lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if max_iter > 5:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10,15],gamma=0.1)
    model.train()
    # train_loss = 0 #to be used later, don't use it yet
    # correct = 0
    # total = 0
    acc_log = []
    for i in range(max_iter):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc_log.append(test(model))
    
    return acc_log

#main
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model_names = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet161', 'densenet169']
model_names = [args.model]
for model_name in model_names:
    model = models.create_model(model_name)

    ##analyze loaded model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters of original model: {pytorch_total_params}')

    if device.type == 'cuda':
        testloader = dataset.create_testset(args.dataset)
        trainloader = dataset.create_trainset(args.dataset,args.batch_size)
        pre_acc = test(model)
        print(f'original acc = {pre_acc}')

    ##measure inference time
    if device.type == 'cuda':
        avg_total_time, avg_time_per_data, std_total_time, std_time_per_data = measure_time(model,20)
        print('Original model inference time')
        print(np.mean(avg_total_time),np.std(std_total_time))
        print(np.mean(avg_time_per_data),np.std(std_time_per_data))

    model = model.cpu()
    decomp_model = tucker_decomposition.decomp_model(model)
    print(decomp_model)

    if device.type == 'cuda':
        decomp_acc = test(decomp_model)
        print(f'Decomp model accuracy: {decomp_acc}')

    ##fine_tune
    if args.fine_tune:
        #trainloader = dataset.create_trainset(args.dataset,args.batch_size)
        acc_log = fine_tune(decomp_model,args.lr,args.epoch)
        ft_decomp_acc = test(decomp_model)
        print(f'Decomp model accuracy after fine tuning: {ft_decomp_acc}')

    if args.save:
        state = {
            'model': decomp_model.state_dict(),
            'decom_method': 'tucker',
        }
        if not os.path.isdir('decomposed_model'):
            os.mkdir('decomposed_model')
        torch.save(state, f'./decomposed_model/tucker_model_{model_name}.pth')
        
        log_dict = {'acc_log': acc_log}
        with open(f'./decomposed_model/ft_log_{model_name}.json', 'w') as outfile:
            json.dump(log_dict, outfile)

    new_pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters of decomp model: {new_pytorch_total_params}')

    if device.type == 'cuda':
        avg_total_time, avg_time_per_data, std_total_time, std_time_per_data = measure_time(decomp_model,20)
        print('Decomp model inference time')
        print(np.mean(avg_total_time),np.std(std_total_time))
        print(np.mean(avg_time_per_data),np.std(std_time_per_data))
