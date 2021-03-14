'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class decomp_LeNet(nn.Module):
    def __init__(self):
        super(decomp_LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,1,1,bias=False),nn.Conv2d(1,3,5,bias=False),nn.Conv2d(3,6,1))
        self.conv2 = nn.Sequential(nn.Conv2d(6,2,1,bias=False),nn.Conv2d(2,5,5,bias=False),nn.Conv2d(5,16,1))
        self.fc1   = nn.Sequential(nn.Linear(400,24,bias=False),nn.Linear(24,120))
        self.fc2   = nn.Sequential(nn.Linear(120,4,bias=False),nn.Linear(4,84))
        self.fc3   = nn.Sequential(nn.Linear(84,7,bias=False),nn.Linear(7,10))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out