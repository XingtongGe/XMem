import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *
gpu_num = torch.cuda.device_count()

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        # print("unload params : ", pretrained_dict.keys() - model_dict.keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print("load params : ", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0



class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        self.addres = False
        self.Encoder = FeatureEncoder()
        self.Decoder = FeatureDecoder()
        self.aligner = PCD_Align()

        self.L1_OffsetPriorEncoder = OffsetPriorEncodeNet(1)
        self.L1_OffsetPriorDecoder = OffsetPriorDecodeNet(1)
        self.L1_BitEstimator_offsetf = NIPS18nocBitEstimator()
        self.L1_BitEstimator_offsetz = ICLR17BitEstimator(out_channel_mv)

        self.resEncoder = ResEncodeNet()
        self.resDecoder = ResDecodeNet()
        self.resPriorEncoder = ResPriorEncodeNet()
        self.resPriorDecoder = ResPriorDecodeNet()
        self.bitEstimator_residualf = NIPS18nocBitEstimator()
        self.bitEstimator_residualz = ICLR17BitEstimator(out_channel_resN)

    def Trainall(self):
        for p in self.parameters():
            p.requires_grad = True

    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def Encode(self, image): 
        return self.Encoder(image)

    def Decode(self, feature):
        return self.Decoder(feature)#.clamp(0., 1.)
    
    # 特征域运动估计和补偿
    def AlignMultiFeature(self, ref_feature, input_feature): 
        # 将decoded_L1_offset添加并return
        L1_fea, enL1_offset, q_L1_offset, decoded_L1_offset = self.aligner(ref_feature, input_feature)
        encoded_L1_offset_prior = self.L1_OffsetPriorEncoder(enL1_offset)
        q_L1_offset_prior = self.Q(encoded_L1_offset_prior)
        decoded_L1_offset_prior = self.L1_OffsetPriorDecoder(q_L1_offset_prior)

        L1_bits_offsetf, _ = self.L1_BitEstimator_offsetf(q_L1_offset, decoded_L1_offset_prior)
        L1_bits_offsetz, _ = self.L1_BitEstimator_offsetz(q_L1_offset_prior)

        bits_offsetf = L1_bits_offsetf
        bits_offsetz = L1_bits_offsetz


        return L1_fea, bits_offsetf, bits_offsetz, decoded_L1_offset
    

    def AddResidual(self, input_feature, aligned_feature):
        encoded_residual = self.resEncoder(input_feature - aligned_feature)

        q_encoded_residual = self.Q(encoded_residual)
        output_feature = aligned_feature + self.resDecoder(q_encoded_residual)

        encoded_residual_prior = self.resPriorEncoder(encoded_residual)
        q_encoded_residual_prior = self.Q(encoded_residual_prior)
        decoded_residual_prior = self.resPriorDecoder(q_encoded_residual_prior)

        bits_residualf, _ = self.bitEstimator_residualf(q_encoded_residual, decoded_residual_prior)
        bits_residualz, _ = self.bitEstimator_residualz(q_encoded_residual_prior)

        return bits_residualf, bits_residualz, output_feature

    def GetLoss(self, input_image, recon_image, aligned_image, bits_offsetf, bits_offsetz, bits_residualf, bits_residualz):
        mse_loss = torch.mean((recon_image - input_image).pow(2))
        aligned_loss = torch.mean((aligned_image - input_image).pow(2))
        im_shape = input_image.size()
        allarea = im_shape[0] * im_shape[2] * im_shape[3]

        bpp_offsetf = bits_offsetf / allarea
        bpp_offsetz = bits_offsetz / allarea
        bpp_residualf = bits_residualf / allarea
        bpp_residualz = bits_residualz / allarea
        bpp = bpp_offsetf + bpp_offsetz + bpp_residualf + bpp_residualz

        return mse_loss, aligned_loss, bpp, bpp_offsetf, bpp_offsetz, bpp_residualf, bpp_residualz
    
    # def foward(self, input, ref):
if __name__ == '__main__':
    import time
    m = VideoCompressor().cuda()
    x = torch.zeros([1, 3, 256, 256]).cuda()
    ref_feature = m.Encode(x)
    t = time.time()
    n = 1
    with torch.no_grad():
        for _ in range(n):
            input_feature = m.Encode(x)
            print('input feature: ',input_feature.shape)
            aligned_feature0, bits_offsetf0, bits_offsetz0, decoded_L1_offset = m.AlignMultiFeature(ref_feature, input_feature)
            # machine_feaure = m.CompensationForMachine(ref_feature, decoded_L1_offset)

            print('aligned_feature0: ',aligned_feature0.shape)
            m.AddResidual(input_feature, aligned_feature0)
            ref_feature = m.Encode(x)
    print(f'{(time.time() - t) / n * 1000} ms')
    # in_channels = [256, 512, 1024]
    # backbone = YOLOPAFPN(in_channels=in_channels)
    # head = YOLOXHead(num_classes=1, in_channels=in_channels)
    # yoloxmodel = YOLOX(backbone, head).cuda()
    # yoloxmodel.eval()
    # output = yoloxmodel(x)
    # print(output.shape)

