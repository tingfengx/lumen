"""
lumen/models/vgg.py
Author (modified by): Tingfeng Xia

VGG network and VGG-LIKE network declaration. 

Including VGG11, 13, 16, 19 models (vanilla from paper). These models contain
five max pooling layers, which is too much for our training data. 

VGGLIKE is a class of models that has only four max-pooling layers. 

Reference : https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# four pooling layers
cfg_vgg_like = {
    'VGGSM': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
}

# five pooling layers
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGLIKE(nn.Module):
    def __init__(self, vgg_name):
        super(VGGLIKE, self).__init__()
        ###############################
        assert vgg_name in cfg_vgg_like
        ###############################
        self.features = self._make_layers(cfg_vgg_like[vgg_name])
        self.classifier = nn.Linear(1024, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


'''VGG11/13/16/19 in Pytorch.'''


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        ###############################
        assert vgg_name in cfg_vgg
        ###############################
        self.features = self._make_layers(cfg_vgg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
