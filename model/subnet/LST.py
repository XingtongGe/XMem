# Latent Space Transform
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import *
from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
try:
    from .dcn.deform_conv import ModulatedDeformConvPack as DCN
    from .dcn.deform_conv import DeformConvPack as DCNv1
except ImportError:
    raise ImportError('Failed to import DCN module.')

# 这里要用的残差块和FVC yolo中的各不相同，故单独定义出来
# 带BN，ksize=3

class ResBlockBN(nn.Module):

    def __init__(self, inputchannel, kernel_size=3, stride=1):
        super().__init__()
        self.layer1 = BaseConv(
            inputchannel, inputchannel, ksize=kernel_size, stride=stride, act="lrelu"
        )
        self.layer2 = BaseConv(
            inputchannel, inputchannel, ksize=kernel_size, stride=stride, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class ResBlockBNUpSample(nn.Module):

    def __init__(self, inputchannel, outputchannel, kernel_size=3, stride=1):
        super().__init__()
        self.layer1 = BaseConv(
            inputchannel, outputchannel, ksize=kernel_size, stride=stride, act="lrelu"
        )
        self.layer2 = BaseConv(
            outputchannel, outputchannel, ksize=kernel_size, stride=1, act="lrelu"
        )
        self.conv = nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=stride, padding=1, groups=1, bias=False)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return self.conv(x) + out

class LkResBlockUpSample(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size=3, stride=1):
        super(LkResBlockUpSample, self).__init__()
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, 1, padding=kernel_size//2)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride=stride, padding=1)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer
        
class ResblocksBN(nn.Module):

    def __init__(self, nf=128, ks=3):
        super(ResblocksBN, self).__init__()
        self.res1 = ResBlockBN(nf)
        self.res2 = ResBlockBN(nf)
        self.res3 = ResBlockBN(nf)

    def forward(self, x):
        return x + self.res3(self.res2(self.res1(x)))

class ResidualBlockLRU(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=128, ks=3):
        super(ResidualBlockLRU, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, ks, 1, ks//2, bias=True)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(nf, nf, ks, 1, ks//2, bias=True)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return identity + out
    
class ResblocksLRU(nn.Module):

    def __init__(self, nf=128, ks=3):
        super(ResblocksLRU, self).__init__()
        self.res1 = ResidualBlockLRU(nf)
        self.res2 = ResidualBlockLRU(nf)
        self.res3 = ResidualBlockLRU(nf)

    def forward(self, x):
        return x + self.res3(self.res2(self.res1(x)))

class LatentSpaceTransform(nn.Module):
    '''
    latent space transform
    带有BN的残差块堆叠
    return: torch.Size([batchSize, 256, H/4, W/4])
    '''
    def __init__(self, inputchannel=32, outputchannel=256) -> None:
        super().__init__()
        self.rb1 = ResBlockBN(inputchannel)
        self.rbU1 = ResBlockBNUpSample(inputchannel, 64)
        self.rb2 = ResBlockBN(64)
        self.rbU2 = ResBlockBNUpSample(64, outputchannel, stride=2)
        self.rb3 = ResBlockBN(outputchannel)
        self.rbU3 = ResBlockBNUpSample(outputchannel, outputchannel, stride=2)
        self.conv = nn.Conv2d(outputchannel, outputchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    
    def forward(self, x):
        x = self.rbU1(self.rb1(x))
        x = self.rbU2(self.rb2(x))
        x = self.rbU3(self.rb3(x))
        x = self.conv(x)
        return x

class LatentMotionTransform(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(64, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        # self.rb3 = ResBlockBN(outputchannel)
        # self.rbU3 = ResBlockBNUpSample(outputchannel, outputchannel, stride=2)
        # self.conv = nn.Conv2d(outputchannel, outputchannel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 再通过resblock进行映射
        x = self.rbU1(self.rb1(L1_fea))
        x = self.rbU2(self.rb2(x))

        return x

# 对encoded motion feature [1,128,16,16]取一半通道 [1,64,16,16]，写一个新的LST，LST还要包括运动补偿操作，随后产生对应的[1,256,32,32]维度特征
class LatentMotionTransformWoBN(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = Resblocks(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = Resblocks(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.rb1 = LkResBlock(nf, nf, 3)
        self.rbU1 = LkResBlockUpSample(nf, 128, 3, stride=2)
        self.rb2 = LkResBlock(128, 128, 3)
        self.rbU2 = LkResBlockUpSample(128, 256, 3, stride=2)

        self.rb3 = LkResBlock(256, 256, 3)
        self.rb4 = LkResBlock(256, 256, 3)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 再通过resblock进行映射
        x = self.rbU1(self.rb1(L1_fea))
        x = self.rbU2(self.rb2(x))
        x = self.rb4(self.rb3(x))
        x = self.conv(x)
        return x
    
class LatentMotionTransform96(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)

        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 再通过resblock进行映射
        x = self.rbU1(self.rb1(L1_fea))
        x = self.rbU2(self.rb2(x))
        x = self.rb4(self.rb3(x))
        x = self.conv(x)
        return x

class LatentMotionTransformwoRef(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)

        self.rb1 = ResBlockBN(128)
        self.rbU1 = ResBlockBNUpSample(128, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)

        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)

        # 再通过resblock进行映射
        x = self.rbU1(self.rb1(L1_offset))
        x = self.rbU2(self.rb2(x))
        x = self.rb4(self.rb3(x))
        x = self.conv(x)
        return x

class LatentMotionTransformAddRes(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.residualDeconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes1 = ResblocksBN(128)
        self.residualDeconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes2 = ResblocksBN(128)
        self.residualDeconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)

        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature, residual_feature):
        # 对x进行处理
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        # 参考帧与motion结合
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 得到预测帧与残差信息相加 再通过resblock进行映射
        res_feature = self.residualRes1(self.residualDeconv1(residual_feature))
        res_feature = self.residualRes2(self.residualDeconv2(res_feature))
        res_feature = self.residualDeconv3(res_feature)
        
        L1_fea = L1_fea + res_feature

        L1_fea = self.rbU1(self.rb1(L1_fea))
        L1_fea = self.rbU2(self.rb2(L1_fea))
        L1_fea = self.rb4(self.rb3(L1_fea))
        L1_fea = self.conv(L1_fea)
        return L1_fea

class LatentMotionTransformAddRes32(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(96, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.residualDeconv1 = nn.ConvTranspose2d(32, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes1 = ResblocksBN(128)
        self.residualDeconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes2 = ResblocksBN(128)
        self.residualDeconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)

        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature, residual_feature):
        # 对x进行处理
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        # 参考帧与motion结合
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 得到预测帧与残差信息相加 再通过resblock进行映射
        res_feature = self.residualRes1(self.residualDeconv1(residual_feature))
        res_feature = self.residualRes2(self.residualDeconv2(res_feature))
        res_feature = self.residualDeconv3(res_feature)
        
        L1_fea = L1_fea + res_feature

        L1_fea = self.rbU1(self.rb1(L1_fea))
        L1_fea = self.rbU2(self.rb2(L1_fea))
        L1_fea = self.rb4(self.rb3(L1_fea))
        L1_fea = self.conv(L1_fea)
        return L1_fea

class LatentMotionTransformAddRes12832(nn.Module):
    def __init__(self, nf=64) -> None:
        super().__init__()
        # 上采样三次，和ref feature进行可变卷积后再通过resblock映射过去
        self.deconv1 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksBN(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksBN(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.L1_dcnpack = DCNv1(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.residualDeconv1 = nn.ConvTranspose2d(32, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes1 = ResblocksBN(128)
        self.residualDeconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.residualRes2 = ResblocksBN(128)
        self.residualDeconv3 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)

        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, x, ref_feature, residual_feature):
        # 对x进行处理
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        L1_offset = self.deconv3(x)
        # 参考帧与motion结合
        L1_fea = self.L1_dcnpack([ref_feature, L1_offset])
        offset = torch.cat([L1_fea, ref_feature], dim=1)
        
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = L1_fea + offset
        # 得到预测帧与残差信息相加 再通过resblock进行映射
        res_feature = self.residualRes1(self.residualDeconv1(residual_feature))
        res_feature = self.residualRes2(self.residualDeconv2(res_feature))
        res_feature = self.residualDeconv3(res_feature)
        
        L1_fea = L1_fea + res_feature

        L1_fea = self.rbU1(self.rb1(L1_fea))
        L1_fea = self.rbU2(self.rb2(L1_fea))
        L1_fea = self.rb4(self.rb3(L1_fea))
        L1_fea = self.conv(L1_fea)
        return L1_fea

class BaseDeConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride=2, padding = 1, output_padding=1
    ):
        super().__init__()
        # same padding
        # pad = (ksize - 1) // 2
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class FeatureReconDecoder(nn.Module):
    '''
    input: [batch, 128, h/4, w/4]
    output: [batch, 3, h, w]
    '''
    def __init__(self, nf=128) -> None:
        super().__init__()
        self.deconv1 = BaseDeConv(256, 128, 3, stride=2, padding = 1, output_padding=1)
        self.res11 = ResLayer(128)
        self.res12 = ResLayer(128)
        self.deconv2 = BaseDeConv(128, 64, 3, stride=2, padding = 1, output_padding=1)
        self.res21 = ResLayer(64)
        self.res22 = ResLayer(64)
        self.deconv3 = BaseDeConv(64, 32, 3, stride=2, padding = 1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding = 1)

    def forward(self, x):
        x = self.deconv1(x)
        # print(x.shape)
        x = self.res12(self.res11(x))
        x = self.deconv2(x)
        # print(x.shape)
        x = self.res22(self.res21(x))
        x = self.deconv3(x)
        x = self.deconv4(x)  
        return x

from .UNet import UNet

class LkResblocks(nn.Module):

    def __init__(self, nf=128, ks=3):
        super(Resblocks, self).__init__()
        self.res1 = LkResBlock(nf, nf, ks)
        self.res2 = LkResBlock(nf, nf, ks)
        self.res3 = LkResBlock(nf, nf, ks)

    def forward(self, x):
        return x + self.res3(self.res2(self.res1(x)))
    
class ScalableResidualTransformNet(nn.Module):
    def __init__(self, nf=64):
        super(ScalableResidualTransformNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(96, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res1 = ResblocksLRU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res2 = ResblocksLRU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_resN, out_channel_M, 5, stride=2, padding=2, output_padding=1)
        self.unet = UNet()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.rb1 = ResBlockBN(nf)
        self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        self.rb2 = ResBlockBN(128)
        self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        self.rb3 = ResBlockBN(256)
        self.rb4 = ResBlockBN(256)
        self.rb5 = ResBlockBN(256)
        self.rb6 = ResBlockBN(256)
        self.rb7 = ResBlockBN(256)
        self.rb8 = ResBlockBN(256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    def forward(self, aligned_feature, q_encoded_residual):
        
        res_feature = self.res1(self.deconv1(q_encoded_residual))
        res_feature = self.res2(self.deconv2(res_feature))
        res_feature = self.deconv3(res_feature)
        machine_recon_feature = aligned_feature + res_feature
        machine_feature = self.unet(machine_recon_feature)
        
        # offset = torch.cat([feature, ref_fea], dim=1)
        # offset = self.lrelu(self.cas_offset_conv1(offset))
        # offset = self.lrelu(self.cas_offset_conv2(offset))
        # feature = feature + offset
        # print('feature: ',feature.shape)
        machine_feature = self.rbU1(self.rb1(machine_feature))
        machine_feature = self.rbU2(self.rb2(machine_feature))
        machine_feature = self.rb4(self.rb3(machine_feature))
        machine_feature = self.rb8(self.rb7(machine_feature))
        machine_feature = self.conv(machine_feature)
        # feature = self.unet(feature)

        return machine_feature, machine_recon_feature #.clamp(0., 1.)

class ScalableAlignTransformNet(nn.Module):
    def __init__(self, nf=64):
        super(ScalableAlignTransformNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res1 = Resblocks()
        self.deconv2 = nn.ConvTranspose2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res2 = Resblocks()
        self.deconv3 = nn.ConvTranspose2d(out_channel_resN, out_channel_M, 5, stride=2, padding=2, output_padding=1)

        # self.unet = UNet()
        # self.rb1 = ResBlockBN(nf)
        # self.rbU1 = ResBlockBNUpSample(nf, 128, stride=2)
        # self.rb2 = ResBlockBN(128)
        # self.rbU2 = ResBlockBNUpSample(128, 256, stride=2)
        # self.rb3 = ResBlockBN(256)
        # self.rb4 = ResBlockBN(256)
        # self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, aligned_feature, q_encoded_residual):
        x = self.res1(self.deconv1(q_encoded_residual))
        x = self.res2(self.deconv2(x))
        x = self.deconv3(x)
        return x + aligned_feature
        # machine_feature = self.unet(aligned_feature)
        
        # machine_feature = self.rbU1(self.rb1(machine_feature))
        # machine_feature = self.rbU2(self.rb2(machine_feature))
        # machine_feature = self.rb4(self.rb3(machine_feature))
        # machine_feature = self.conv(machine_feature)

        # return machine_feature
    
class ScalableResidualTransformNetStem(nn.Module):
    def __init__(self, nf=64):
        super(ScalableResidualTransformNetStem, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(96, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res1 = Resblocks()
        self.deconv2 = nn.ConvTranspose2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res2 = Resblocks()
        self.deconv3 = nn.ConvTranspose2d(out_channel_resN, out_channel_M, 5, stride=2, padding=2, output_padding=1)
        self.unet = UNet()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.rb1 = ResBlockBN(nf)
        # self.rbU1 = ResBlockBNUpSample(nf, 128)
        self.rb2 = ResBlockBN(nf)
        self.rb3 = ResBlockBN(nf)
        # self.rb4 = ResBlockBN(nf)
        # self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    def forward(self, aligned_feature, q_encoded_residual):
        
        res_feature = self.res1(self.deconv1(q_encoded_residual))
        res_feature = self.res2(self.deconv2(res_feature))
        res_feature = self.deconv3(res_feature)
        machine_recon_feature = aligned_feature + res_feature
        machine_feature = self.unet(machine_recon_feature)
        # offset = torch.cat([feature, ref_fea], dim=1)
        # offset = self.lrelu(self.cas_offset_conv1(offset))
        # offset = self.lrelu(self.cas_offset_conv2(offset))
        # feature = feature + offset
        # print('feature: ',feature.shape)
        machine_feature = self.rb2(self.rb1(machine_feature))
        machine_feature = self.rb3(machine_feature)

        return machine_feature, machine_recon_feature #.clamp(0., 1.)

if __name__ == '__main__':
    import time
    # model = LatentMotionTransformWoBN().cuda()
    # x = torch.zeros([1, 96, 16, 16]).cuda()
    # ref_feature = torch.zeros([1, 64, 128, 128]).cuda()
    # t = time.time()
    # n = 1
    # with torch.no_grad():
    #     output = model(x, ref_feature)
    #     print(output.shape)
    # print(f'{(time.time() - t) / n * 1000} ms')
    model = FeatureReconDecoder().cuda()
    x = torch.zeros([1, 256, 32, 32]).cuda()
    output = model(x)
    print(output.shape)