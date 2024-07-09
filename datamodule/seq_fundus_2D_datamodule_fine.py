# -*- coding:utf-8 -*-
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from natsort import natsorted
from torch.utils.data.dataset import Dataset
import cv2 as cv
from torchvision import transforms
import numpy as np


def SeqFundusDatamodule(
        data_root,
        image_size=(256, 256),
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        mode='train',
):

    dataset = SIGF_Dataset(data_root=data_root, image_size=image_size, mode=mode)

    output_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return output_dataloader


class SIGF_Dataset(Dataset):
    def __init__(self, data_root, image_size, mode):

        self.data_root = data_root
        self.mode = mode
        self.image_size = image_size
        
        self.data_list = []
        
        if self.mode == 'train':
            path1 = os.path.join(self.data_root, 'train')
            datalist1 = natsorted(os.listdir(path1))
            for dataname in datalist1:
                self.data_list.append(os.path.join(path1, dataname))
            path2 = os.path.join(self.data_root, 'test')
            datalist2 = natsorted(os.listdir(path2))
            for dataname in datalist2:
                self.data_list.append(os.path.join(path2, dataname))
        else:
            path2 = os.path.join(self.data_root, 'test')
            datalist2 = natsorted(os.listdir(path2))
            for dataname in datalist2:
                self.data_list.append(os.path.join(path2, dataname))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_path = self.data_list[idx]
        data_id = data_path.split('/')[-1]
        data = np.load(data_path)

        fundus_images = data['seq_imgs']
        fundus_images = torch.Tensor(np.array(fundus_images))
        fundus_images = fundus_images / 255
        fundus_images = torch.permute(fundus_images, (0, 3, 1, 2))
        fundus_images = torch.cat((fundus_images[:, 2:3, :, :], fundus_images[:, 1:2, :, :], fundus_images[:, 0:1, :, :]), dim=1)
        
        real_times = torch.LongTensor(np.array(data['times']))
        
        image_label = torch.LongTensor(np.array(data['labels']))

        coarse = np.load('results/vqldm/vqldm_2d_test_2024-01-17T22-40-15/features_2/'+data_id)
        coarse_data = torch.Tensor(np.array(coarse['z_0']))
        
        return {
            'image': fundus_images,
            'time': real_times,
            'image_id': data_id,
            'label': image_label,
            'coarse': coarse_data,
        }
