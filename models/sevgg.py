"""
https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


"""
lumen/models/sevgg.py
Author (modified by): Tingfeng Xia

VGG network with Squeeze and Excitation

Reference: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch 
Reference: https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/
"""


# four pooling layers
cfg_sevgg = {
    'SEVGG1': [64, 'M', 128, 'M', 256, 'SE', 256, 'M', 512, 'SE', 512, 'M'],
}


class SEVGG(nn.Module):
    def __init__(self, sevgg_name):
        super(SEVGG, self).__init__()
        ###############################
        assert sevgg_name in cfg_sevgg
        ###############################
        self.features = self._make_layers(cfg_sevgg[sevgg_name])
        self.classifier = nn.Linear(1024, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'SE':
                layers += [SELayer(channels)]
            else:
                layers += [
                    nn.Conv2d(channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
