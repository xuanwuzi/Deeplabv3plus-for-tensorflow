# -*- coding: UTF-8 -*-

"""
样本随机分组（训练集：验证集：测试集 = 8：1：1）
"""
import os
import shutil
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    print('样本随机分组...')
    labels_path = r'data\train\label'
    images_path = r'data\train\img'
    yangben_path = r'data\yangben'

    labels_ = [i for i in os.listdir(labels_path)]

    train_labels_, val_test_labels_ = train_test_split(labels_,
                                                       test_size=0.2,
                                                       random_state=5)
    val_labels_, test_labels_ = train_test_split(val_test_labels_,
                                                 test_size=0.5,
                                                 random_state=5)
    print('训练集:', len(train_labels_))
    print('验证集:', len(val_labels_))
    print('测试集:', len(test_labels_))
    print('正在划分数据集...')

    train_images = yangben_path + '\\train_images'
    train_labels = yangben_path + '\\train_labels'
    val_images = yangben_path + '\\val_images'
    val_labels = yangben_path + '\\val_labels'
    test_images = yangben_path + '\\test_images'
    test_labels = yangben_path + '\\test_labels'

    # 创建输出文件夹
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)

    for i in train_labels_:
        shutil.copyfile(labels_path+'\\'+i, train_labels+'\\'+i)
        shutil.copyfile(images_path+'\\'+i, train_images+'\\'+i)
    for i in val_labels_:
        shutil.copyfile(labels_path+'\\'+i, val_labels+'\\'+i)
        shutil.copyfile(images_path+'\\'+i, val_images+'\\'+i)
    for i in test_labels_:
        shutil.copyfile(labels_path+'\\'+i, test_labels+'\\'+i)
        shutil.copyfile(images_path+'\\'+i, test_images+'\\'+i)

    print('DONE!')
