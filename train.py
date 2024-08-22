# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from nets.deeplab import Deeplabv3
from osgeo import gdal
import tensorflow as tf


def load_image(_image_path):
    """定义图像加载函数"""

    # 打开TIFF文件
    dataset = gdal.Open(_image_path, gdal.GA_ReadOnly)
    # 读取波段数据到数组
    image = dataset.ReadAsArray()
    image = image.astype(int)
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, :3]

    # 异常值处理
    image[image < 0] = 0
    image[image > 255] = 0
    # 规格化至0-1之间
    image = image/255.

    # image = image[0:16, 0:16, :]

    # 关闭数据集
    dataset = None

    return image


def load_label(_label_path):
    """定义标签加载函数"""

    # 打开TIFF文件
    label_data = gdal.Open(_label_path, gdal.GA_ReadOnly)
    # 读取波段数据到数组
    label = label_data.ReadAsArray()
    label = label.astype(int)

    label[label > 255] = 0
    label[label <= 0] = 0
    label[label >= 1] = 1

    # label = label[0:16, 0:16]

    del label_data

    return label


def generator(_image_paths, _label_paths, _batch_size):
    """迭代器加载函数"""
    while 1:
        images = []
        labels = []
        for i in range(len(_image_paths)):
            image = load_image(_image_paths[i])
            label = load_label(_label_paths[i])
            images.append(image)
            labels.append(label)
            if len(images) % _batch_size == 0:
                x = np.array(images)
                y = np.array(labels)

                yield x, y
                images = []
                labels = []


@tf.keras.utils.register_keras_serializable()
def loss_a(y_true, y_predict):
    y_true_f = tf.cast(y_true, tf.float32)
    y_predict_f = tf.cast(y_predict, tf.float32)
    yz = 0.5
    y_predict_f = tf.where(y_predict_f <= yz, 0., y_predict_f)
    y_predict_f = tf.where(y_predict_f > yz, 1., y_predict_f)

    _loss1 = dice_loss(y_true, y_predict)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    _loss2 = bce(y_true_f, y_predict_f)

    _loss = 0.5*_loss1 + 0.5*_loss2

    return _loss


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_predict):

    smooth = 1e-5
    y_true_f = tf.cast(y_true, tf.float32)
    y_predict_f = tf.cast(y_predict, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_predict_f)
    _loss = 1. - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_predict_f) + smooth))

    return _loss


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=None, name=None, dtype=None):
        super(CustomMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_predict, sample_weight=None):
        # 计算MIoU
        y_true_f = tf.cast(y_true, tf.float32)
        y_predict_f = tf.cast(y_predict, tf.float32)
        yz = 0.5
        y_predict_f = tf.where(y_predict_f <= yz, 0., y_predict_f)
        y_predict_f = tf.where(y_predict_f > yz, 1., y_predict_f)

        return super().update_state(y_true_f, y_predict_f, sample_weight)


if __name__ == '__main__':

    print('train...')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 输入图像的形状
    input_shape = (512, 512, 3)
    # input_shape = (16, 16, 3)

    # 学习率
    lr = 0.0001
    # lr = 0.0004
    # lr = 0.005

    # 批次
    batch_size = 2
    # 轮数
    epochs = 30
    # 样本文件夹路径
    data_dir = r'data\yangben'
    # 最后模型保存路径
    models_path = r'Deeplabv3_512'

    # 创建输出文件夹
    os.makedirs(models_path, exist_ok=True)
    model_path = models_path + '\\' + models_path + '.h5'

    # 加载训练数据
    train_image_files = sorted(os.listdir(os.path.join(data_dir, 'train_images')))
    train_label_files = sorted(os.listdir(os.path.join(data_dir, 'train_labels')))
    train_image_paths = [os.path.join(data_dir, 'train_images', filename) for filename in
                         train_image_files if filename.split('.')[-1] == 'png']
    train_label_paths = [os.path.join(data_dir, 'train_labels', filename) for filename in
                         train_label_files if filename.split('.')[-1] == 'png']

    # 加载验证数据
    validation_image_files = sorted(os.listdir(os.path.join(data_dir, 'val_images')))
    validation_label_files = sorted(os.listdir(os.path.join(data_dir, 'val_labels')))
    validation_image_paths = [os.path.join(data_dir, 'val_images', filename) for filename in
                              validation_image_files if filename.split('.')[-1] == 'png']
    validation_label_paths = [os.path.join(data_dir, 'val_labels', filename) for filename in
                              validation_label_files if filename.split('.')[-1] == 'png']

    # 构建迭代器
    train_generator = generator(train_image_paths, train_label_paths,
                                batch_size)
    validation_generator = generator(validation_image_paths, validation_label_paths,
                                     batch_size)

    # 加载模型
    # model = Deeplabv3(input_shape=input_shape, num_classes=2, backbone="mobilenet")
    model = Deeplabv3(input_shape=input_shape, num_classes=2, backbone="xception")

    # 自定义度量mIoU
    custom_mIoU_metric = CustomMeanIoU(num_classes=2, name='mIoU')

    # 编译模型
    # model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr), loss=dice_loss, metrics=['accuracy'])
    model.compile(optimizer=Adam(lr), loss=loss_a, metrics=[custom_mIoU_metric])

    # 打印模型结构
    model.summary()

    # 创建一个ModelCheckpoint回调
    checkpoint = ModelCheckpoint(model_path.split('.')[0]+'_epoch_{epoch:02d}.h5',  # 模型文件名中包含轮次号
                                 save_best_only=False,  # 如果为True，则只保存在验证集上性能最好的模型
                                 save_weights_only=True,  # 如果为True，则只保存模型权重而不保存整个模型
                                 save_freq=1  # 控制保存模型的频率，每多少个轮次保存一次
                                 )
    # 训练模型
    model.fit(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=validation_generator,
              steps_per_epoch=len(train_image_files)//batch_size,
              validation_steps=len(validation_image_files)//batch_size,
              verbose=1,
              callbacks=[checkpoint]
             )

    # 保存模型
    model.save(model_path)

    print('Done！')
