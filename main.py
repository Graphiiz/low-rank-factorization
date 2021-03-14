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
import lenet
import decomposition


parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
#train,test arguments
parser.add_argument('--train', action='store_true',
                    help='train mode')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
parser.add_argument('--gamma', default=0.5, type=int, help='gamma for learning rate scheduler')
parser.add_argument('--step-size', default=50, type=int, dest='step_size',help='gamma for learning rate scheduler')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--test', action='store_true', help='test mode, model is required')
parser.add_argument('--test-decomp', dest='test_decomp', action='store_true', help='test decomp model mode, decomp model is required')
#parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--model', '-m', type=str,
                    help='path to saved model or .pth file')
#decomposition arguments
parser.add_argument('--decomp', '-d', action='store_true',
                    help='do rank decomposition, model is required')
parser.add_argument("--type", dest="decompose_type", default="tucker", 
                    choices=["tucker", "cp"], #original is choices=["tucker", "cp", "channel", "depthwise", "spatial"]
                    help="type of decomposition, if None then no decomposition")
parser.add_argument("-r", "--rank", dest="rank", type=int, default=None,
                    help="use pre-specified rank for all layers")
parser.add_argument("--conv-ranks", dest="conv_ranks", nargs='+', type=int, default=None,
                    help="a list of ranks specifying rank for each convolution layer")                    
parser.add_argument("--exclude-first-conv", dest="exclude_first_conv", action="store_true",
                    help="avoid decomposing first convolution layer")
parser.add_argument("--exclude-linears", dest="exclude_linears", action="store_true",
                    help="avoid decomposing fully connected layers")
#fine-tuning arguments
parser.add_argument('--fine-tuning', '-ft', dest='fine_tuning',action='store_true', #dest='xxx' means store value in args.xxx
                    help='do fine-tuning, model is required')



args = parser.parse_args()

best_acc = 0

#function

#train
def train(epoch):
    model.to(device)
    model.train()
    train_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'finish epoch #{epoch}')
    print(f'Training accuracy = {correct/total}')
    
def test(epoch):
    global best_acc #declare this allow you make changes to global variable
    model.eval()
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
    print(f'Test accuracy = {correct/total}')
    acc = correct/len(testloader)
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

            
#main
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model is not None:
    if (args.test or args.decomp) == False:
        print('test mode or decomp mode is required')
        exit(0)
if args.train:
    print('Create model...')
    model = lenet.LeNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start training the model...')
    print('==> Preparing data..')
    trainloader = dataset.train_loader()
    testloader = dataset.test_loader()
    print('==> Datasets are ready')
    num_epoch = args.epoch
    for epoch in range(num_epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

if args.test:
    if args.model is None: 
        print('.pth file of pretrained model is required')
        exit(0)
    print('Load model...')
    info_dict = torch.load(args.model) #see format of .pth file in train function
    model = lenet.LeNet()
    model.load_state_dict(info_dict['model'])
    print('load model successfully')
    model.to(device)
    model.eval()
    print('Start testing the model...')
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
    print(f'Test accuracy = {correct/len(testloader)}')

if args.decomp:
    if args.model is None: 
        print('.pth file of pretrained model is required')
        exit(0)
    print('Load model...')
    info_dict = torch.load(args.model,map_location=torch.device('cpu')) #see format of .pth file in train function
    loaded_model = lenet.LeNet() # ti make it general when we use different models
    loaded_model.load_state_dict(info_dict['model'])
    print('load model successfully')
    loaded_model.eval()
    loaded_model.cpu()
    if args.decompose_type == 'tucker':
        rank = [2,2] #for tucker, rank is the number input and output channel of R3/R4 defined by [2,2] for example.
        ranks = None
        decomp_config = {"criterion": None,"threshold": None,"rank": rank, "exclude_first_conv": False, "exclude_linears": False, "conv_ranks": ranks, "mask_conv_layers": None}
        decomp_model = decomposition.decompose_model(loaded_model, 'tucker', decomp_config)
        state = {
            'model': decomp_model.state_dict(),
            'decom_method': 'tucker',
        }
        if not os.path.isdir('decomposed_model'):
            os.mkdir('decomposed_model')
        torch.save(state, './decomposed_model/tucker_model.pth')
    elif args.decompose_type == 'cp':
        rank = None
        ranks = None
        decomp_config = {"criterion": None,"threshold": None,"rank": rank, "exclude_first_conv": False, "exclude_linears": False, "conv_ranks": ranks, "mask_conv_layers": None}
        decomp_model = decomposition.decompose_model(loaded_model, 'cp', decomp_config)
        state = {
            'model': decomp_model.state_dict(),
            'decom_method': 'cp',
        }
        if not os.path.isdir('decomposed_model'):
            os.mkdir('decomposed_model')
        torch.save(state, './decomposed_model/cp_model.pth')
    else:
        print('Invalid arguments: "tucker" or "cp" required')
    
if args.fine_tuning:
    print('load decomp model...')
    info_dict = torch.load(args.model) #see format of .pth file in train function
    model = lenet.decomp_LeNet()
    model.load_state_dict(info_dict['model'])
    print('load model successfully')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start fine-tuning the model...')
    print('==> Preparing data..')
    trainloader = dataset.train_loader()
    testloader = dataset.test_loader()
    print('==> Datasets are ready')
    num_epoch = args.epoch
    for epoch in range(num_epoch):
        train(epoch)
        #scheduler.step()

if args.test_decomp:
    if args.model is None: 
        print('.pth file of pretrained model is required')
        exit(0)
    print('Load model...')
    info_dict = torch.load(args.model) #see format of .pth file in train function
    model = lenet.decom_LeNet()
    model.load_state_dict(info_dict['model'])
    print('load model successfully')
    model.to(device)
    model.eval()
    print('Start testing the model...')
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
    print(f'Test accuracy = {correct/len(testloader)}')

    

