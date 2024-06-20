import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from torchvision import models


def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


# Neural Network
class Network(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(Network, self).__init__()
    
        Cin,Hin,Win=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]
        self.num_points=params["num_points"]
        self.dropout_rate=params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, self.num_points * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X)); X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X));X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X));X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        X = X.view(-1, self.num_points, 2)
        return self.sigmoid(X)  # sigmoid or points between -1 and 1


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x



class Resi(nn.Module):
    def __init__(self, params):
        super(Resi, self).__init__()
        encoder = models.resnet34(pretrained=True)
        # make the encoder regression model
        encoder.fc = nn.Linear(512, params['num_points']*2)
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.params = params
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.params['num_points'], 2)
        return self.sigmoid(x)


class Resi2(nn.Module):
    def __init__(self, params):
        super(Resi2, self).__init__()
        encoder = models.resnet34(pretrained=True)
        self.encoder = encoder
        conv_seq = nn.Sequential(
            nn.Conv2d(512,64,kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder.avgpool = conv_seq
        self.encoder.fc = Identity()
        self.encoder.fc = nn.Linear(64 * 7 * 7,512)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512, params['num_points']*2)
        self.sigmoid = nn.Sigmoid()
        self.params = params
    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = x.view(-1, self.params['num_points'], 2)
        x = self.sigmoid(x)
        return x


class ResiBig(nn.Module):  # It achieves worse results in general than Resi2
    def __init__(self, params):
        super(ResiBig, self).__init__()
        encoder = models.resnet34(pretrained=True)
        # make the encoder regression model
        # encoder.fc = nn.Linear(512, params['num_points']*2)
        self.encoder = encoder
        conv_seq = nn.Sequential(
            nn.Conv2d(512,64,kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder.avgpool = conv_seq
        self.encoder.fc = Identity()
        self.encoder.fc = nn.Linear(64 * 7 * 7,1568)  # more than 512
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1568, params['num_points']*2)
        self.sigmoid = nn.Sigmoid()
        self.params = params
    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = x.view(-1, self.params['num_points'], 2)
        x = self.sigmoid(x)
        return x


class Mobi(nn.Module):
    def __init__(self, params):
        super(Mobi, self).__init__()
        encoder = models.mobilenet_v2(pretrained=True)
        # make the encoder regression model
        encoder.classifier[1] = nn.Linear(1280, params['num_points']*2)
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.params = params
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.params['num_points'], 2)
        return self.sigmoid(x)


class Mobi2(nn.Module):
    def __init__(self, params):
        super(Mobi2, self).__init__()
        encoder = models.mobilenet_v2(pretrained=True)
        # make the encoder regression model
        # encoder.classifier[1] = nn.Linear(1280, params['num_points']*2)
        self.encoder = encoder
        classi_seq = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(1280, 512),
            nn.ReLU()
        )
        self.encoder.classifier = classi_seq
        self.linear1 = nn.Linear(512, params['num_points']*2)
        self.sigmoid = nn.Sigmoid()
        self.params = params
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        x = x.view(-1, self.params['num_points'], 2)
        x = self.sigmoid(x)
        return x
