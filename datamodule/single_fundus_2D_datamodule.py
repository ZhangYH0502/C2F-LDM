import torch
from torch.utils.data import DataLoader
import os
from natsort import natsorted
from torch.utils.data.dataset import Dataset
import cv2 as cv
import numpy as np


def SingleFundusDatamodule(
        data_root,
        image_size=(256, 256),
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
):

    dataset = SIGF_Dataset(data_root=data_root, image_size=image_size)

    output_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return output_dataloader


class SIGF_Dataset(Dataset):
    def __init__(self, data_root, image_size):

        self.data_root = data_root
        self.data_list = natsorted(os.listdir(self.data_root))
        self.image_size = image_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_id = self.data_list[idx]
        fundus_image = cv.imread(os.path.join(self.data_root, data_id), cv.IMREAD_COLOR)
        fundus_image = cv.resize(fundus_image, (self.image_size[0], self.image_size[1]), interpolation=cv.INTER_CUBIC)
        fundus_image = np.array(fundus_image)
        fundus_image = fundus_image / 255
        fundus_image = torch.Tensor(fundus_image)
        fundus_image = torch.permute(fundus_image, (2, 0, 1))

        return {
            'image': fundus_image,
            'image_id': data_id,
        }
