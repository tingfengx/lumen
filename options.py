"""
lumen/options.py
Author: Tingfeng Xia

Training options specification class. 
"""


import torch


class Opt():
    def __init__(
        self,
        model="VGGSM",
        dataset="FER2013",
        bs=128,
        lr=.01,
        usecuda=torch.cuda.is_available(),
        lr_decay_rate=0.9,
        lr_decay_start=40,
        lr_decay_every=5,
        total_epoch=200,
        # random cut
        cut_size=(20, 44),
    ):
        self.model = model
        self.dataset = dataset
        self.bs = bs
        self.lr = lr
        self.usecuda = usecuda
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_start = lr_decay_start
        self.lr_decay_every = lr_decay_every
        self.total_epoch = total_epoch
        self.cut_size = cut_size
        self.savepath = "_".join([self.dataset, self.model])

    def __str__(self):
        return f"""
model = {self.model}
dataset = {self.dataset}
batchsize = {self.bs}
learning rate = {self.lr}
device = {"cuda" if self.usecuda else "cpu"}
lr decay:
    start = {self.lr_decay_start}
    rate = {self.lr_decay_rate}
    every = {self.lr_decay_every}
total epoch = {self.total_epoch}
cut size = {self.cut_size}
model save path = {self.savepath}
        """
