import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.module_util as mutil
# import modules.Unet_common as common
import functools
import config as c



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, nf=c.nf, gc=c.gc, bias=True, use_snorm=False):

        super(ResidualDenseBlock_out, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(32 * 4 + input, output, kernel_size=1, padding=0, stride=1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.attention = ChannelAttention(32*4+input)
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = torch.cat((x, x1, x2, x3, x4),1)
        x6 = self.attention(x5) * x5
        #out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = self.conv5(x6)
        return out
