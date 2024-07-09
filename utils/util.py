import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

import json, os
def save_metrics_to_json(test_metric_dict, global_iter, result_dir, json_name='single_fold_result.json'):
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    json_path = os.path.join(result_dir, json_name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = dict()
    # print(results)
    def write_to_result(results_dict, iter, metric_dict):
        # print(iter, results_dict)
        iter = str(iter)
        if iter not in results_dict.keys(): results_dict[iter] = dict()
        # print(iter, results_dict)
        for k in metric_dict.keys():
            if k not in results_dict[iter]: results_dict[iter][k] = []
            results_dict[iter][k] += metric_dict[k]

    write_to_result(results, global_iter, test_metric_dict)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)



def find_model_by_iter(model_dir, target_iter):
    names = os.listdir(model_dir)
    for name in names:
        if 'model-epoch='+str(target_iter) in name: return os.path.join(model_dir, name)

import cv2 as cv
from natsort import natsorted
import numpy as np
from torchvision import transforms
import torch
def read_cube_to_np(img_dir, stack_axis=2, cvflag =  cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name),cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=stack_axis)
    return imgs

def read_cube_to_tensor(path, stack_axis=1, cvflag =  cv.IMREAD_GRAYSCALE):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cvflag)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=stack_axis)
    return imgs
from torchvision.utils import save_image
def save_cube_from_tensor(img, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for j in range(img.shape[0]):
        img_path = os.path.join(result_dir, str(j + 1) + '.png')
        save_image(img[j, :, :], img_path)

def save_cube_from_numpy(data, result_name, tonpy=False):
    if tonpy:
        np.save(result_name +'.npy', data)
    else:
        result_dir = result_name
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        for i in range(data.shape[0]):
            cv.imwrite(os.path.join(result_dir, str(i+1)+'.png'), data[i, ...])