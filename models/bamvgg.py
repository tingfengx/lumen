"""
lumen/bamvgg.py
Author: Tingfeng Xia

VGG (LIKE) models integerated with BAM (Bottleneeck Attention Modules).

Reference: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch 
"""

import torch
import torch.nn as nn
from models.bam import BAM
import torch.nn.functional as F


# four pooling layers
cfg_bam = {
    'BAMVGG_SM': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
    'BAMVGG_MD': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'BAMVGG_LG': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'BAMVGG_TI': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}


class BAMVGG(nn.Module):
    def __init__(self, name, reduction_ratio=4, dilation=2):
        super(BAMVGG, self).__init__()
        ###############################
        assert name in cfg_bam
        self.reduction_ratio = reduction_ratio
        self.dilation = dilation
        ###############################
        self.features = self._make_layers(cfg_bam[name])
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
                layers += [
                    BAM(channels, reduction_ratio=self.reduction_ratio,
                        dilation=self.dilation),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            else:
                layers += [
                    nn.Conv2d(channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    print(BAMVGG("BAMVGG_SM"))
