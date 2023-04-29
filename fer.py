"""
lumen/fer.py
Author (modified by): Tingfeng Xia

Fer2013 Dataset class. 

Reference : https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch 
"""


from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split  # training set or test set

        # now load the picked numpy arrays
        if self.split == "train":
            self.data = h5py.File(
                './datasets/fer2013/train_processed.h5', 'r', driver='core')
            self.train_data = self.data['pixels']
            self.train_labels = self.data['targets']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((-1, 24, 48))

        elif self.split == "test":
            self.data = h5py.File(
                './datasets/fer2013/test_processed.h5', 'r', driver='core')
            self.test_data = self.data['pixels']
            self.test_labels = self.data['targets']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((-1, 24, 48))

        else:
            print("[FER2013/__init__()] Something went wrong!")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'test':
            return len(self.test_data)
        print("[FER2013/__len__()] Something went wrong!")
        return 0


if __name__ == "__main__":
    fer2013 = FER2013()
    print(fer2013.train_data.shape)
    fer2013 = FER2013(split="test")
    print(fer2013.test_data.shape)
