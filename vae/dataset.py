import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pythae.data.datasets import DatasetOutput



class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = '../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()
        print(type(self.imgs))

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        
        
        return x#DatasetOutput(data=x)
