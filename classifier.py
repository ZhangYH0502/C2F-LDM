import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import argparse
import os
from tqdm import tqdm
import numpy as np

import utils.metrics as um


class ClsDataset(Dataset):
    def __init__(self, img_root, flg):
        self.root = img_root + "/" + flg
        self.data_list = os.listdir(self.root)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_id = self.data_list[idx]

        data = np.load(self.root + "/" + image_id)
        image = torch.Tensor(np.array(data['image']))
        label = torch.LongTensor(np.array(data['label']))

        sample = {}
        sample['image'] = image
        sample['label'] = label
        sample['imageIDs'] = image_id

        return sample


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout=0.1):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')

            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nn.ReLU())
            projection_modulelist.append(nn.BatchNorm1d(mlp_dim))
            projection_modulelist.append(nn.Dropout(dropout))

            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)

        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.last_layer(x)
        return x


def model_run(args, net, train_loader, valid_loader, test_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  # weight_decay=0.0004)
    step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    save_best_model = um.SaveBestModel(args.model_name)

    for epoch in range(1, args.epochs+1):
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))

        train_loader.dataset.epoch = epoch

        all_logits_train, all_preds_train, all_targs_train, all_ids_train, train_loss = run_epoch(net, train_loader, optimizer, 'Training', True)
        print('\n')
        print('output train metric:')
        train_metrics = um.compute_metrics(all_preds_train, all_targs_train, train_loss)

        all_logits_valid, all_preds_valid, all_targs_valid, all_ids_valid, valid_loss = run_epoch(net, valid_loader, None, 'Validating', False)
        print('\n')
        print('output valid metric:')
        valid_metrics = um.compute_metrics(all_preds_valid, all_targs_valid, valid_loss)

        if test_loader is not None:
            all_logits_test, all_preds_test, all_targs_test, all_ids_test, test_loss = run_epoch(net, test_loader, None, 'Testing', False)
            print('\n')
            print('output test metric:')
            test_metrics = um.compute_metrics(all_preds_test, all_targs_test, test_loss)
        else:
            test_metrics = valid_metrics
            test_loss = valid_loss

        step_scheduler.step(valid_loss)

        save_best_model.evaluate(valid_metrics, test_metrics, epoch, net, all_preds_valid, all_ids_valid)


def run_epoch(net, data, optimizer, desc, train=False):
    if train:
        net.train()
        optimizer.zero_grad()
    else:
        net.eval()

    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_targets = []
    all_logits = []
    all_image_ids = []
    loss_total = []

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['image'].float()
        labels = batch['label']

        if train:
            pred = net(images.cuda())
        else:
            with torch.no_grad():
                pred = net(images.cuda())

        loss = criterion(pred, labels.cuda())
        loss_total.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pred_arg = torch.argmax(pred.clone().detach(), dim=1)
        batch_num = labels.shape[0]
        for i in range(batch_num):
            all_logits.append(pred[i, :].data.cpu())
            all_predictions.append(pred_arg[i].data.cpu())
            all_targets.append(labels[i].data.cpu())
        all_image_ids += batch['imageIDs']

    all_logits = torch.stack(all_logits, dim=0)
    all_predictions = torch.stack(all_predictions, dim=0)
    all_targets = torch.stack(all_targets, dim=0)
    loss_total = np.mean(loss_total)

    return all_logits, all_predictions, all_targets, all_image_ids, loss_total


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/research/data')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001) # 0.00001
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=6)
    args = parser.parse_args()

    train_dataset = ClsDataset(img_root=args.data_root, flg="train")
    valid_dataset = ClsDataset(img_root=args.data_root, flg="valid")
    test_dataset = ClsDataset(img_root=args.data_root, flg="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = MLP(input_dim=4, mlp_dims=[512] * args.num_layers + [args.num_labels])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.cuda()

    model_run(args, net, train_loader, valid_loader, test_loader)
