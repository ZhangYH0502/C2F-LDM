import numpy as np
from PIL import Image
import os
import datetime


def read_text(file_name):
    x = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            x.append(line)
    return x


def write_text(file_name, file_content_1=None):
    log_file = open(file_name, "a")
    for c1 in file_content_1:
        log_file.write(c1 + '\n')
    log_file.close()


def compute_time_interval(t1, t2):
    d1 = datetime.date(int(t1[:4]), int(t1[4:6]), int(t1[6:]))
    d2 = datetime.date(int(t2[:4]), int(t2[4:6]), int(t2[6:]))
    time_int = (d2-d1).days / 365
    return round(time_int)


def SIGF_single(mode='train'):
    img_path = '/research/deepeye/zhangyuh/data/SIGF-database/'+mode+'/image'
    lab_path = '/research/deepeye/zhangyuh/data/SIGF-database/'+mode+'/label'
    
    out_path = '/research/deepeye/zhangyuh/data/SIGF-Total-W-Valid/'+mode
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    patients = os.listdir(img_path)

    for patient in patients:

        patient_path = os.path.join(img_path, patient)

        seq_images = os.listdir(patient_path)
        seq_len = len(seq_images)

        # seq_labels = read_text(os.path.join(lab_path, patient + '.txt'))

        for idx in range(seq_len):

            seq_name = seq_images[idx]

            fundus = Image.open(os.path.join(patient_path, seq_name))

            # label = int(seq_labels[idx])

            fundus.save(out_path+'/'+seq_name)


def SIGF(mode='train'):

    max_time = 0
    pos_num = 0
    neg_num = 0

    img_path = '/research/deepeye/zhangyuh/data/SIGF-database/'+mode+'/image'
    lab_path = '/research/deepeye/zhangyuh/data/SIGF-database/'+mode+'/label'
    # polar_path = '/research/deepeye/zhangyuh/data/SIGF-database/final_polar'
    # atten_path = '/research/deepeye/zhangyuh/data/SIGF-database/final_atten'

    if mode == 'test':
        out_path = '/research/deepeye/zhangyuh/data/SIGF_Seq/test/'
    else:
        out_path = '/research/deepeye/zhangyuh/data/SIGF_Seq/train/'

    patients = os.listdir(img_path)

    for patient in patients:

        patient_path = os.path.join(img_path, patient)

        seq_images = os.listdir(patient_path)
        seq_len = len(seq_images)

        seq_labels = read_text(os.path.join(lab_path, patient+'.txt'))

        image_list = []
        time_list = []
        label_list = []
        # polar_list = []
        # atten_list = []

        for idx in range(seq_len):

            seq_name = seq_images[idx]

            fundus = Image.open(os.path.join(patient_path, seq_name))
            fundus = fundus.resize((256, 256), Image.LANCZOS)
            fundus = np.array(fundus, dtype=np.float32)
            image_list.append(fundus)

            seq_name_split = seq_name.split('_')
            seq_time = seq_name_split[1] + seq_name_split[2] + seq_name_split[3]
            time_list.append(seq_time)

            label_list.append(int(seq_labels[idx]))

            # try:
            #     polar = Image.open(os.path.join(polar_path, seq_name[0:-4] + '.jpg'))
            # except:
            #     polar = Image.open(os.path.join(polar_path, seq_name[0:-4] + '.JPG'))
            # polar = polar.resize((224, 224), Image.LANCZOS)
            # polar = np.array(polar, dtype=np.float32)
            # polar_list.append(polar)
            #
            # try:
            #     atten = Image.open(os.path.join(atten_path, seq_name[0:-4] + '.jpg'))
            # except:
            #     atten = Image.open(os.path.join(atten_path, seq_name[0:-4] + '.JPG'))
            # atten = atten.resize((224, 224), Image.LANCZOS)
            # atten = np.array(atten, dtype=np.float32)
            # atten_list.append(atten)

        max_num = len(image_list) - 6 + 1
        for i in range(max_num):

            image_s = image_list[i:i+6]
            time_s = time_list[i:i+6]
            label_s = label_list[i:i+6]
            # polar_s = polar_list[i:i+6]
            # atten_s = atten_list[i:i+6]
            
            if label_s[-1] == 1:
                pos_num = pos_num + 1
            else:
                neg_num = neg_num + 1

            time_s_new = [int(0)]
            for j in range(5):
                time_tmp = compute_time_interval(time_s[0], time_s[j+1])
                time_s_new.append(time_tmp)
            if time_s_new[5] > max_time:
                max_time = time_s_new[5]

            image_s = np.array(image_s)
            label_s = np.array(label_s)
            time_s_new = np.array(time_s_new)
            # polar_s = np.array(polar_s)
            # atten_s = np.array(atten_s)

            out_name = patient + '_' + str(i)

            # np.savez(
            #     out_path + out_name + '.npz',
            #     seq_imgs=image_s,
            #     times=time_s_new,
            #     labels=label_s,
            # )
    
    print(pos_num)
    print(neg_num)


if __name__ == '__main__':

    # SIGF('train')
    SIGF('valid')
    # SIGF('test')

    # path = 'SIGF\\train'
    # patients = os.listdir(path)
    # mint = 10000
    # maxt = 0
    # for patient in patients:
    #     X = np.load(path + '\\' + patient)
    #     A = X['times']
    #     t0 = None
    #     for idx in range(A.shape[0]):
    #         if idx == 0:
    #             t0 = str(A[idx])
    #         else:
    #             ti = str(A[idx])
    #             time_int = compute_time_interval(t0, ti)
    #             if time_int > maxt:
    #                 maxt = time_int
    #             if time_int < mint:
    #                 mint = time_int
    # print(mint)
    # print(maxt)
















