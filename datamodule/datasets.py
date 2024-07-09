import os

import albumentations as A
import cv2 as cv
import torch
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class choroid_labeled_Dataset(Dataset):
    def __init__(self, img_root, label_root, region_root, cube_names, transform):
        assert os.path.exists(img_root)
        assert os.path.exists(label_root)

        self.transform = transform
        self.cube_names = []
        self.img_names = []
        self.img_root = img_root
        self.label_root = label_root
        self.region_root = region_root

        for i, (cube_name) in enumerate(cube_names):
            img_cube_dir = os.path.join(img_root, cube_name)
            assert os.path.exists(img_cube_dir)

            img_names = natsorted(os.listdir(img_cube_dir))
            for i, (A_name) in enumerate(img_names):
                self.cube_names.append(cube_name)
                self.img_names.append(A_name)

    def __getitem__(self, index):
        real_A = cv.imread(os.path.join(self.img_root, self.cube_names[index], self.img_names[index]),
                           cv.IMREAD_GRAYSCALE)
        label = cv.imread(os.path.join(self.label_root, self.cube_names[index], self.img_names[index]),
                          cv.IMREAD_GRAYSCALE)
        region = cv.imread(os.path.join(self.region_root, self.cube_names[index], self.img_names[index]),
                           cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=real_A, masks=[label, region])
            real_A = transformed['image']
            label = transformed['masks'][0]
            region = transformed['masks'][1]
        real_A = transforms.ToTensor()(real_A)
        label = torch.from_numpy(label).long() // 255
        label = label.unsqueeze(0)
        region = torch.from_numpy(region).long() // 255
        region = region.unsqueeze(0)

        return {'image': real_A, 'label': label, 'region': region, 'cube_name': self.cube_names[index],
                'img_name': self.img_names[index]}

    def __len__(self):
        return len(self.img_names)



class unlabeled_choroid_Dataset(Dataset):
    def __init__(self, img_root, region_root, cube_names, transform):
        assert os.path.exists(img_root)

        self.transform = transform
        self.cube_names = []
        self.img_names = []
        self.img_root = img_root
        self.region_root = region_root

        for i, (cube_name) in enumerate(cube_names):
            img_cube_dir = os.path.join(img_root, cube_name)
            assert os.path.exists(img_cube_dir)

            img_names = natsorted(os.listdir(img_cube_dir))
            for i, (A_name) in enumerate(img_names):
                self.cube_names.append(cube_name)
                self.img_names.append(A_name)

    def __getitem__(self, index):
        real_A = cv.imread(os.path.join(self.img_root, self.cube_names[index], self.img_names[index]),
                           cv.IMREAD_GRAYSCALE)
        region = cv.imread(os.path.join(self.region_root, self.cube_names[index], self.img_names[index]),
                           cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=real_A, masks=[region])
            real_A = transformed['image']
            region = transformed['masks'][0]
        real_A = transforms.ToTensor()(real_A)
        region = torch.from_numpy(region).long() // 255
        region = region.unsqueeze(0)

        return {'image': real_A, 'region': region, 'cube_name': self.cube_names[index],
                'img_name': self.img_names[index]}

    def __len__(self):
        return len(self.img_names)

class reconstruct_Dataset(Dataset):
    def __init__(self, img_root, label_root, cube_names, transform):
        assert os.path.exists(img_root)
        assert os.path.exists(label_root)

        self.transform = transform
        self.cube_names = []
        self.img_names = []
        self.img_root = img_root
        self.label_root = label_root

        for i, (cube_name) in enumerate(cube_names):
            img_cube_dir = os.path.join(img_root, cube_name)
            assert os.path.exists(img_cube_dir)

            img_names = natsorted(os.listdir(img_cube_dir))
            for i, (A_name) in enumerate(img_names):
                self.cube_names.append(cube_name)
                self.img_names.append(A_name)

    def __getitem__(self, index):
        real_A = cv.imread(os.path.join(self.img_root, self.cube_names[index], self.img_names[index]),
                           cv.IMREAD_GRAYSCALE)
        label = cv.imread(os.path.join(self.label_root, self.cube_names[index], self.img_names[index]),
                          cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=real_A, masks=[label])
            real_A = transformed['image']
            label = transformed['masks'][0]
        real_A = transforms.ToTensor()(real_A)
        label = transforms.ToTensor()(label)

        return {'image': real_A, 'label': label, 'cube_name': self.cube_names[index],
                'img_name': self.img_names[index]}

    def __len__(self):
        return len(self.img_names)



class ConcatDataset(Dataset):
    def __init__(self, datasets, dataset_names, align=False):
        self.datasets = datasets
        self.transforms = transforms
        self.align = align
        self.dataset_names = dataset_names

    def __getitem__(self, index):
        if self.align:
            data = [d[index] for d in self.datasets]
        else:
            data = {}
            data[self.dataset_names[0]] = self.datasets[0][index]
            for i, d in enumerate(self.datasets):
                if i == 0: continue
                id = torch.randint(0, len(d), (1,))
                data[self.dataset_names[i]] = d[id]
        return data

    def __len__(self):
        return len(self.datasets[0])
