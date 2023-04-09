import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import *
from .offsetcoder import OffsetEncodeNet, OffsetDecodeNet
import yolox
from .UNet import UNet
try:
    from .dcn.deform_conv import ModulatedDeformConvPack as DCN
    from .dcn.deform_conv import DeformConvPack as DCNv1
except ImportError:
    raise ImportError('Failed to import DCN module.')

class LST(nn.Module):
    '''
    latent space transform
    '''
    def __init__(self, nf=out_channel_M):
        super(LST, self).__init__()
        


class CompensationMachineNet(nn.Module):
    def __init__(self, nf=out_channel_M):
        super(CompensationMachineNet, self).__init__()
        # 搞一个DCN和一个UNet，输出的feature输入到resnet tail中
        # 需要解决feature scale的问题
        groups = nf // 8
        self.machine_dcn = DCNv1(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.unet = UNet()
        # self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        # self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, ref_fea, L1_offset):
        
        feature = self.machine_dcn([ref_fea, L1_offset])
        # offset = torch.cat([feature, ref_fea], dim=1)
        # offset = self.lrelu(self.cas_offset_conv1(offset))
        # offset = self.lrelu(self.cas_offset_conv2(offset))
        # feature = feature + offset
        # print('feature: ',feature.shape)
        feature = self.unet(feature)

        return feature #.clamp(0., 1.)