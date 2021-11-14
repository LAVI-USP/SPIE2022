"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import torch 
from torch import nn
from torchvision import models
from collections import namedtuple

class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """
    def __init__(self,  num_filters = 64, inputLayer=False):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResidualBlock, self).__init__()
        
        
        in_filters = num_filters
        if inputLayer:
            in_filters = 1

        self.conv1 = nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet
    
    Source: https://arxiv.org/abs/1512.03385
    """
    def __init__(self, num_filters=64):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResNet, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock(num_filters, inputLayer=True)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)
                
        self.conv_last = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
           
        identity = x
        
        out = self.block1(x)
        
        out = self.block2(out)
                
        out = self.block3(out)
        
        out = self.block4(out)
                
        out = self.conv_last(out)
                
        out = self.relu(out + identity)
        
        return out

class Gen(nn.Module):
    """
    Generator from WGAN
    """
    def __init__(self, num_filters=64):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(Gen, self).__init__()
        
        self.generator = ResNet(num_filters)

    def forward(self, x):

        return self.generator(x)
    
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = torch.cat([X, X, X], dim=1)
        #X = normalize_batch(X)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

