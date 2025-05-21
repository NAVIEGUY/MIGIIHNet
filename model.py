import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet

device = torch.device('cpu')
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


class MC(nn.Module):
    def __init__(self):
        super(MC, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=4, stride=2, padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.BN2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.BN3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.BN4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.BN5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 12, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, stego_x):
        out1 = self.lrelu(self.BN1(self.conv1(stego_x)))
        out2 = self.lrelu(self.BN2(self.conv2(out1)))
        out3 = self.lrelu(self.BN3(self.conv3(out2)))
        out4 = self.lrelu(self.BN4(self.conv4(out3)))
        out5 = self.lrelu(self.BN5(self.conv5(out4 + out2)))
        out = self.conv6(out5 + out1)
        out = out
        return out

class MC_model(nn.Module):
    def __init__(self):
        super(MC_model, self).__init__()

        self.model = MC()

    def forward(self, stego_x):
        out1 = self.model(stego_x)
        out = out1
        return out

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)




c