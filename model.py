# import required packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, is_bias = True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=is_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=is_bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=is_bias)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=is_bias)
        self.fc1 = nn.Linear(4096, 50, bias=is_bias)
        self.fc2 = nn.Linear(50, 10, bias=is_bias)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 26 x 26 x 32 RF3
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 12 x 12 X 64 RF6
        x = F.relu(self.conv3(x), 2) # 10 x 10 x 128 RF10
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 4 x 4 x 256 RF16
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x # CE loss will take care of log_softmax


# GAP implementation
class NetGAP(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, is_bias = True):
        super(NetGAP, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=is_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=is_bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=is_bias)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=is_bias)

        self.fc1 = nn.Linear(256, 50, bias=is_bias)
        self.fc2 = nn.Linear(50, 10, bias=is_bias)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 26 x 26 x 32 RF3
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 12 x 12 X 64 RF6
        x = F.relu(self.conv3(x), 2) # 10 x 10 x 128 RF10
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 4 x 4 x 256 RF15
        # x = x.view(-1, 4096)
        # Global average pooling
        x = F.avg_pool2d(x, kernel_size = 4).squeeze()

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x # CE loss will take care of log_softmax

# antMan and GAP implementation
class NetAntmanGAP(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, is_bias = True):
        super(NetAntmanGAP, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=is_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=is_bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=is_bias)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=is_bias)
        ## the antman kernel
        self.antman = nn.Conv2d(256, 10, kernel_size=1, bias=is_bias)

        # self.fc1 = nn.Linear(256, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 26 x 26 x 32
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 12 x 12 X 64
        x = F.relu(self.conv3(x), 2) # 10 x 10 x 128
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 4 x 4 x 256
        x = self.antman(x) # 4 x 4 x 10
        # Global average pooling
        x = F.avg_pool2d(x, kernel_size = 4).squeeze()
        # x = self.ap(x).squeeze()
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x # CE loss will take care of log_softmax
    
