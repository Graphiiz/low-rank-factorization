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

#import relative .py files
import dataset
import model
import decomposition


parser = argparse.ArgumentParser(description='PyTorch CIFAR10')

parser.add_argument('--model_path', type=str, help='path to model')
parser.add_argument('--model', type=str, help='model type eg. Lenet, VGG16')
parser.add_argument('--dataset', default=None, type=str,help='dataset choices = ["MNIST","CIFAR10"]')
parser.add_argument('--batch_size', default=128, type=int,help='batch size')
parser.add_argument('--fine_tune', action='store_true',help='do fine tuning')
parser.add_argument('--save', action='store_true',help='save decomp model')
parser.add_argument('--epoch', type=int, default=5, help='epoch for fine tuning')

args = parser.parse_args()


def measure_time(model,repetition):
    model.to(device)
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

    model.cpu()
    
    final_time_per_dataset = np.mean(time_per_dataset)
    final_time_per_data = np.mean(time_per_data)
    std_time_per_dataset = np.std(time_per_dataset)
    std_time_per_data = np.std(time_per_data)

    return final_time_per_dataset, final_time_per_data, std_time_per_dataset, std_time_per_data

def test(model):
    model.to(device)
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
    model.to('cpu')
    return correct/total

def fine_tune(model,lr=0.001, max_iter=5):
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr = lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    # train_loss = 0 #to be used later, don't use it yet
    # correct = 0
    # total = 0
    for i in range(max_iter):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
  
#main
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
info_dict = torch.load(args.model_path,map_location=torch.device('cpu')) #see format of .pth file in train function
model = model.create_model(args.model) # ti make it general when we use different models
model.load_state_dict(info_dict['model'])

##analyze loaded model
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'total parameters of original model: {pytorch_total_params}')

testloader = dataset.create_testset(args.dataset)
test(model)

##measure inference time
avg_total_time, avg_time_per_data, std_total_time, std_time_per_data = measure_time(model,10)
print('Original model inference time')
print(np.mean(avg_total_time),np.std(std_total_time))
print(np.mean(avg_time_per_data),np.std(std_time_per_data))

##do tucker decomposition
rank = None #for tucker, rank is the number input and output channel of R3/R4 defined by [2,2] for example. None = use VBMF approximation.
ranks = None
decomp_config = {"criterion": None,"threshold": None,"rank": rank, "exclude_first_conv": False, "exclude_linears": False, "conv_ranks": ranks, "mask_conv_layers": None}
decomp_model = decomposition.decompose_model(model, 'tucker', decomp_config)

decomp_acc = test(decomp_model)
print(f'Decomp model accuracy: {decomp_acc}')

##fine_tune
if args.fine_tune:
    trainloader = dataset.create_trainset(args.dataset,args.batch_size)
    fine_tune(decomp_model,args.epoch)
    ft_decomp_acc = test(decomp_model)
    print(f'Decomp model accuracy after fine tuning: {ft_decomp_acc}')

if args.save:
    state = {
        'model': decomp_model.state_dict(),
        'decom_method': 'tucker',
    }
    if not os.path.isdir('decomposed_model'):
        os.mkdir('decomposed_model')
    torch.save(state, f'./decomposed_model/tucker_model_{args.model}.pth')

new_pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'total parameters of decomp model: {new_pytorch_total_params}')

avg_total_time, avg_time_per_data, std_total_time, std_time_per_data = measure_time(decomp_model,10)
print('Decomp model inference time')
print(np.mean(avg_total_time),np.std(std_total_time))
print(np.mean(avg_time_per_data),np.std(std_time_per_data))








