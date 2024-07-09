# -*- coding:utf-8 -*-
# basic 2d datasets
import os
import cv2 as cv
from torch.utils.data.dataset import Dataset
from natsort import natsorted
from torchvision import transforms
import torch
import numpy as np
def read_cube(path):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cv.IMREAD_GRAYSCALE)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=1)
    print(imgs.shape)
    return imgs

def check_interpolation(imgs, target_cube_size):
    if target_cube_size[-1] != imgs.shape[-1] or target_cube_size[-2] != imgs.shape[-2] or target_cube_size[-3] != \
            imgs.shape[-3]:
        imgs = imgs.unsqueeze(0)
        # print('interpolation', target_cube_size, imgs.shape)
        imgs = torch.nn.functional.interpolate(imgs, size=target_cube_size, mode='trilinear', align_corners=True)
        imgs = imgs.squeeze(0)
    return imgs


def augmentation_torch_multi(images):
    #horizontal flip
    if torch.randint(0, 2, (1,)).item()==0:
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(1,))
    #Vertical flip
    if torch.randint(0, 2, (1,)).item()==0:
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(3,))
    #rot90
    if torch.randint(0, 2, (1,)).item()==0:
        k=torch.randint(1, 4, (1,)).item()
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(3,))
            images[i]=torch.rot90(images[i],k,dims=(1,3))
    for i in range(len(images)):
        images[i] = (images[i]-0.5)*2
        images[i] = images[i].detach()
    return images

class Opt3D_multi_cat_Dataset(Dataset):
    def __init__(self, data_roots, cube_names= None, with_path = False, with_aug = False, cube_size=(200,320,200)):
        self.data_name_lists = []
        self.data_path_lists = []
        for data_root in data_roots:
            assert os.path.exists(data_root)
            if cube_names is None:
                cube_names = natsorted(os.listdir(data_root))
            cube_pathes = [os.path.join(data_root, name) for name in cube_names]
            self.data_name_lists.append(cube_names)
            self.data_path_lists.append(cube_pathes)
        self.cube_size = cube_size
        self.with_path = with_path
        self.with_aug = with_aug

    def __getitem__(self, index):
        imgs_list = []
        for cube_paths in self.data_path_lists:
            imgs = read_cube(cube_paths[index])
            imgs = check_interpolation(imgs, self.cube_size)
            imgs_list.append(imgs)

        if self.with_aug:
            imgs_list = augmentation_torch_multi(imgs_list)

        if self.with_path:
            return imgs_list, self.data_name_lists[0][index]
        else:
            return imgs_list
    def __len__(self):
        return len(self.data_name_lists[0])



class Opt3D_multi_npy_Dataset(Dataset):
    def __init__(self, data_roots, cube_names= None, with_path = False, with_aug = False, cube_size=(200,320,200)):
        self.data_name_lists = []
        self.data_path_lists = []
        for data_root in data_roots:
            assert os.path.exists(data_root)
            if cube_names is None:
                cube_names = natsorted(os.listdir(data_root))
            cube_pathes = [os.path.join(data_root, name) for name in cube_names]
            self.data_name_lists.append(cube_names)
            self.data_path_lists.append(cube_pathes)
        self.cube_size = cube_size
        self.with_path = with_path
        self.with_aug = with_aug

    def __getitem__(self, index):
        imgs_list = []
        for cube_paths in self.data_path_lists:
            imgs = np.load(cube_paths[index])
            imgs = torch.from_numpy(imgs).float()
            imgs /= 255.
            imgs = imgs.unsqueeze(0)
            imgs = check_interpolation(imgs, self.cube_size)
            imgs -= 0.5
            imgs /= 0.5
            imgs_list.append(imgs)

        if self.with_path:
            return imgs_list, self.data_name_lists[0][index]
        else:
            return imgs_list
    def __len__(self):
        return len(self.data_name_lists[0])