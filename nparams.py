import loaders
import numpy as np
from models.vgg import VGGLIKE, VGG
from models.sevgg import SEVGG16, SEVGG8, SEVGG32
from models.bamvgg import BAMVGG

cfg_vgg_like = {
    'VGGLIKE_SM': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
    'VGGLIKE_MD': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGGLIKE_LG': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'VGGLIKE_TI': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}

for name in cfg_vgg_like:
    model = VGGLIKE(name)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(name, params)

# four pooling layers
cfg_sevgg16 = {
    'SEVGG16_SM': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
    'SEVGG16_MD': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'SEVGG16_LG': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'SEVGG16_TI': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}

for name in cfg_sevgg16:
    model = SEVGG16(name)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(name, params)

cfg_bam = {
    'BAMVGG_SM': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
    'BAMVGG_MD': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'BAMVGG_LG': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'BAMVGG_TI': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}

for name in cfg_bam:
    model = BAMVGG(name)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(name, params)