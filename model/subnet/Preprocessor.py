import torch 
import torch.nn as nn
from .UNetImage import UNetImage

class Preprocessor(nn.Module):

    def __init__(self):
        super().__init__()
        # 上层分支
        self.conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        # 下层分支
        self.UNet = UNetImage()
    
    def forward(self,x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.UNet(x)
        return x2+x3

def build_model():
    pro = Preprocessor()
    # print("build success!")
    x = torch.randn(4,3,16,16)
    out = pro(x)
    print(out.shape)

if __name__ == '__main__':
    build_model()