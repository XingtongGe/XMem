import torch.nn as nn
from .basics import *


class FeatureEncoder(nn.Module):
    '''
    Feature Encoder
    '''
    def __init__(self, nf=out_channel_M):
        super(FeatureEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, 5, 2, 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.feature_extraction = Resblocks(nf)

    def forward(self, x):
        x = self.conv1(x)
        return self.feature_extraction(x)



class FeatureDecoder(nn.Module):
    '''
    Feature Decoder
    '''
    def __init__(self, nf=out_channel_M, back_RBs=10):
        super(FeatureDecoder, self).__init__()
        self.recon_trunk = Resblocks(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.deconv1 = nn.ConvTranspose2d(nf, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.recon_trunk(x)
        x = self.deconv1(x)
        return x