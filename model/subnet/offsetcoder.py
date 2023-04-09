import math
import torch.nn as nn
import torch
from .basics import *



class OffsetEncodeNet(nn.Module):
    '''
    Compress offset
    '''
    def __init__(self, inp=out_channel_M):
        super(OffsetEncodeNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_mv, 5, stride=2, padding=2)
        self.res1 = Resblocks(out_channel_mv)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2)
        self.res2 = Resblocks(out_channel_mv)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.res1(self.conv1(x))
        x = self.res2(self.conv2(x))
        return self.conv3(x)
        

class OffsetDecodeNet(nn.Module):
    '''
    Decompress offset
    '''
    def __init__(self, out=out_channel_M):
        super(OffsetDecodeNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2, output_padding=1)
        self.res1 = Resblocks(out_channel_mv)
        self.deconv2 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2, output_padding=1)
        self.res2 = Resblocks(out_channel_mv)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv, out_channel_M, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        x = self.deconv3(x)
        return x

class OffsetPriorEncodeNet(nn.Module):
    '''
    Offset Prior Encode Net
    '''
    def __init__(self, level=1):
        super(OffsetPriorEncodeNet, self).__init__()
        if level == 1:
            stride2 = stride3 = 2
        elif level == 2:
            stride2 = 1
            stride3 = 2
        else:
            stride2 = stride3 = 1
        self.conv1 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=stride2, padding=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=stride3, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)

class OffsetPriorDecodeNet(nn.Module):
    '''
    Offset Prior Decode Net
    '''
    def __init__(self, level=1):
        if level == 1:
            stride1 = stride2 = 2
            outputp1 = outputp2 = 1
        elif level == 2:
            stride1 = 2
            stride2 = 1
            outputp1 = 1
            outputp2 = 0
        else:
            stride1 = stride2 = 1
            outputp1 = outputp2 = 0
        super(OffsetPriorDecodeNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 5, stride=stride1, padding=2, output_padding=outputp1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv2 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv * 3 // 2, 5, stride=stride2, padding=2, output_padding=outputp2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv * 3 // 2, out_channel_mv * 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x)
