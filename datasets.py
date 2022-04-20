import sys
sys.path.append('../')
import os.path as osp
import os
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
import imageio


def img_loader(path, input_size=None):
    img = np.array(imageio.imread(path))
    img = img.astype(np.float32) / 255
    return img

class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

class Data_model_view(data.Dataset):
    def __init__(self, paf_path, filelist_fp, transform=None, **kargs):
        self.paf_path = paf_path
        self.filelist_fp = filelist_fp
        self.transform = transform
        self.filelist = []
        lines = Path(filelist_fp).read_text().strip().split('\n')
        for line in lines:
            self.filelist.append(line)

        self.img_loader = img_loader


    def __getitem__(self, index):
        filename= self.filelist[index]

        paf = self.img_loader(osp.join(self.paf_path, filename + '.jpg'))
        vis_map = self.img_loader(osp.join(self.paf_path, filename + '_vis.jpg'))
        vis_map = vis_map[:,:,None]

        shapemap = self.img_loader(osp.join(self.paf_path, filename + '_shape.jpg'))

        if self.transform is not None:
            paf = self.transform(paf)
            vis_map = self.transform(vis_map)
            shapemap = self.transform(shapemap)
        return paf, vis_map, shapemap, filename

    def __len__(self):
        return len(self.filelist)

