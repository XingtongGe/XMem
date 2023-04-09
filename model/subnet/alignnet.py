import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import *
from .offsetcoder import OffsetEncodeNet, OffsetDecodeNet

try:
    from .dcn.deform_conv import ModulatedDeformConvPack as DCN
    from .dcn.deform_conv import DeformConvPack as DCNv1
except ImportError:
    raise ImportError('Failed to import DCN module.')


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


class PCD_Align(nn.Module):
    '''
    Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=out_channel_M, groups=8, compressoffset=True):
        super(PCD_Align, self).__init__()
        groups = nf // 8
        self.compressoffset = compressoffset

        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_offset_encoder = OffsetEncodeNet()
        self.L1_offset_decoder = OffsetDecodeNet()
        self.L1_dcnpack = DCNv1(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def forward(self, ref_fea, inp_fea):
        # L1
        L1_offset = torch.cat([ref_fea, inp_fea], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        enL1_offset = self.L1_offset_encoder(L1_offset)
        q_L1_offset = self.Q(enL1_offset)
        if self.compressoffset:
            L1_offset = self.L1_offset_decoder(q_L1_offset)
        else:
            L1_offset = self.L1_offset_decoder(enL1_offset)
        L1_fea = self.L1_dcnpack([ref_fea, L1_offset])
        # print('ref_fea: ',ref_fea.shape)
        # print('L1_offset: ',L1_offset.shape)
        # print('L1_fea: ',L1_fea.shape)
        offset = torch.cat([L1_fea, ref_fea], dim=1)

        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset

        return L1_fea, enL1_offset, q_L1_offset, L1_offset
