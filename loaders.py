"""
lumen/loaders.py
Author: Tingfeng Xia

Pytorch dataset loaders (train and test) for FER2013, processed 
to contain 24 x 48 pixels around the eye portion. Then randomly
cropped to 20 x 44 for the sake of model training. 

Test loader retains the center 20 x 44 pixels. 

Pytorch model loaders. Initialize VGGLIKE model with correct 
arguments. 
"""


from transforms import transforms
from options import Opt
import torch
from fer import FER2013
from models.vgg import VGGLIKE


def initialize_train_loader_with_opt(opt):
    # train set transformation
    transform_train = transforms.Compose([
        transforms.RandomCrop(opt.cut_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # prepare train loader
    trainset = FER2013(split='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.bs, shuffle=True, num_workers=1)

    return trainloader


def initialize_test_loader_with_opt(opt):
    # test set transformation
    transform_test = transforms.Compose([
        transforms.TenCrop(opt.cut_size),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
    ])

    # prepare test loader
    testset = FER2013(split='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.bs, shuffle=False, num_workers=1)

    return testloader


def initialize_model_with_opt(opt: Opt):
    return VGGLIKE(opt.model)
