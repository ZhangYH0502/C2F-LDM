import torch
import numpy as np
import csv

import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import matplotlib.pyplot as plt
import sklearn.metrics as sm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(preds, targs):
    cm = sm.confusion_matrix(targs, preds, labels=None)
    plt.matshow(cm, cmap=plt.cm.Oranges)
    plt.colorbar()
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[j, i] >= 30:  # 自己设定变为字体变白的阈值
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', color='w')
            else:
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', color='k')

    plt.tick_params(labelsize=8)  # 设置类别名称的字体大小
    # 设置x轴和y轴的标题
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # 设置x轴和y轴的标题的字体
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # 设置类别名称
    plt.xticks(range(0, 5), labels=['Urgent', 'Semi', 'Non', 'Observe', 'SeeDoc'])
    plt.yticks(range(0, 5), labels=['Urgent', 'Semi', 'Non', 'Observe', 'SeeDoc'])
    # 保存图片，设置保存图片的名称和格式
    plt.savefig('cm2.png', format='png', dpi=1600, bbox_inches='tight')


def compute_tpr_fpr(y_true, y_pred):
    fpr, tpr, threshold = sm.roc_curve(y_true, y_pred, pos_label=1)
    youden_index = np.argmax(tpr - fpr)
    thr = threshold[youden_index]

    preds_thr = y_pred.copy()
    preds_thr[preds_thr < thr] = 0
    preds_thr[preds_thr >= thr] = 1

    tp = np.sum(y_true * preds_thr).astype('float32')
    fp = np.sum((1 - y_true) * preds_thr).astype('float32')

    tn = np.sum((1 - y_true) * (1 - preds_thr)).astype('float32')
    fn = np.sum(y_true * (1 - preds_thr)).astype('float32')

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def compute_metrics(outputs, epoch):
    preds = outputs['preds'].numpy()
    preds = np.argmax(preds, axis=1)

    targs = outputs['targs'].numpy()

    accuracy = sm.accuracy_score(y_true=targs, y_pred=preds)
    sensitivity, specificity = compute_tpr_fpr(y_true=targs, y_pred=preds)
    auc = sm.roc_auc_score(y_true=targs, y_score=preds)

    print('loss: {:0.3f}'.format(outputs['loss']))
    print('accuracy: {:0.1f}'.format(accuracy * 100))
    print('sensitivity: {:0.1f}'.format(sensitivity * 100))
    print('specificity: {:0.1f}'.format(specificity * 100))
    print('auc: {:0.1f}'.format(auc * 100))

    metrics_dict = {}
    metrics_dict['epoch'] = epoch
    metrics_dict['loss'] = round(outputs['loss'], 4)
    metrics_dict['accuracy'] = round(accuracy * 100, 2)
    metrics_dict['sensitivity'] = round(sensitivity * 100, 2)
    metrics_dict['specificity'] = round(specificity * 100, 2)
    metrics_dict['auc'] = round(auc * 100, 2)

    return metrics_dict


class WriteLog:
    def __init__(self, model_name):
        self.model_name = model_name
        open(model_name+'/train.csv', "w").close()
        open(model_name+'/valid.csv', "w").close()

    def log_losses(self, file_name, metrics):
        with open(self.model_name+'/'+file_name, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            output = [str(metrics['epoch']),
                      str(metrics['loss']),
                      str(metrics['accuracy']),
                      str(metrics['sensitivity']),
                      str(metrics['specificity']),
                      str(metrics['auc'])]
            writer.writerow(output)


class SaveBestModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.best = 100000000

    def evaluate(self, valid_metrics, model, valid_outputs, is_write_img, mode=''):
        valid_result = valid_metrics['loss']
        if valid_result < self.best:
            self.best = valid_result

            print('> Saving Model\n')
            save_dict = {'epoch': valid_metrics['epoch'],
                         'state_dict': model.state_dict(),
                         'loss': valid_metrics['loss'],
                         # 'accuracy': valid_metrics['accuracy'],
                         # 'sensitivity:': valid_metrics['sensitivity'],
                         # 'specificity:': valid_metrics['specificity'],
                         # 'auc:': valid_metrics['auc'],
                         }
            torch.save(save_dict, self.model_name + '/best_model.pt')

            if is_write_img:
                outpath = self.model_name + '/' + mode + '/'
                write_imgs(valid_outputs, outpath)

            print('\n')
            print('best loss:  {:0.1f}'.format(valid_result * 100))


def write_imgs(valid_outputs, out_path=''):
    path1 = out_path + 'preds/'
    path2 = out_path + 'targs/'
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
        targ_flg = False
    else:
        targ_flg = True
    rects = valid_outputs['rects']
    imgs = valid_outputs['imgs']
    # ids = valid_outputs['ids']
    for idx in range(rects.shape[0]):
        write_pred = rects[idx, :, :, :] * 255
        write_pred = write_pred.numpy().astype(np.uint8)
        write_pred[write_pred > 255] = 255
        write_pred[write_pred < 0] = 0
        write_pred = np.transpose(write_pred, (1, 2, 0))
        write_pred = Image.fromarray(write_pred)
        write_pred.save(path1 + str(idx) + '.png')
        # write_pred.save(path1 + ids[idx] + '.png')

        if targ_flg is False:
            write_img = imgs[idx, :, :, :] * 255
            write_img = write_img.numpy().astype(np.uint8)
            write_img[write_img > 255] = 255
            write_img[write_img < 0] = 0
            write_img = np.transpose(write_img, (1, 2, 0))
            write_img = Image.fromarray(write_img)
            write_img.save(path2 + str(idx) + '.png')
            # write_img.save(path2 + ids[idx] + '.png')


if __name__ == "__main__":

    metrics_dict = {}
    metrics_dict['epoch'] = 1
    metrics_dict['loss'] = 111
    metrics_dict['accuracy'] = 111
    metrics_dict['precision'] = 111
    metrics_dict['recall'] = 111
    metrics_dict['f1'] = 111
    logg = WriteLog('results')
    logg.log_losses('train.csv', metrics_dict)
    logg.log_losses('train.csv', metrics_dict)







