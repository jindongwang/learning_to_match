import numpy as np
import pickle
import os
import glob
import shutil

data_folder = '/data/jindwang/data/all_data'

def generate_label_txt():
    with open(os.path.join(data_folder, 'COVID-DA-master/data/COVID-19_task_for_python3.pkl'), 'rb') as f:
        train_dict = pickle.load(f)
    train_list_labeled = train_dict['train_list_labeled'] # labeled data (train sub-directory)
    train_list_unlabeled = train_dict['train_list_unlabeled'] # unlabeled data (train sub-directory)
    val_list = train_dict['val_list'] # val sub-directory
    test_list = train_dict['test_list'] # test sub-directory
    fp = open('train_list_covid19_labeled.txt', 'w')
    for item in train_list_labeled:
        fp.write(item[0] + ',' + str(item[1]) + '\n')
    fpun = open('train_list_covid19_unlabeled.txt', 'w')
    for item in train_list_unlabeled:
        fpun.write(item[0] + ',' + str(item[1]) + '\n')
    fp2 = open('val_list_covid19.txt', 'w')
    for item in val_list:
        fp2.write(item[0] + ',' + str(item[1]) + '\n')
    fptest = open('test_list_covid19.txt', 'w')
    for item in test_list:
        fptest.write(item[0] + ',' + str(item[1]) + '\n')

def gen_class_folder():
    SOURCE_ONLY = {
    'train': 'train_list_pneumonia.txt',
    'valid': 'val_list_pneumonia.txt',
    'test': 'test_list_covid19.txt'}
    files_src = os.listdir(data_folder + '/all_data_pneumonia/train')
    # os.makedirs('/home/jindwang/mine/data/pneumonia/0/')
    # os.makedirs('home/jindwang/mine/data/pneumonia/1/')
    # os.makedirs('/home/jindwang/mine/data/covid/0/')
    # os.makedirs('home/jindwang/mine/data/covid/1/')
    labels = open(os.path.join(data_folder, 'labels', SOURCE_ONLY['test'])).readlines()
    cnt = 0
    for item in labels:
        line = item.strip().split(',')
        img_name = line[0]
        label = int(line[1])
        if label == 0:
            shutil.copyfile(os.path.join(data_folder, 'all_data_covid/test', img_name), os.path.join('/home/jindwang/mine/data/covid/0', img_name))
        else:
            shutil.copyfile(os.path.join(data_folder, 'all_data_covid/test', img_name), os.path.join('/home/jindwang/mine/data/covid/1', img_name))
        cnt += 1
        print(cnt)


if __name__ == "__main__":
    gen_class_folder()