# -*- coding: UTF-8 -*-

"""
二分类精度评价
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix


def read_tiff_as_array(tiff_path):
    """
    读取TIFF图像并将其读取为NumPy数组。

    参数：
    tiff_path (str) - TIFF图像文件的路径

    返回：
    image_array (numpy.ndarray) - 读取的图像数据作为NumPy数组
    """
    try:
        # 打开TIFF文件
        dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)

        if dataset is None:
            raise Exception("无法打开TIFF文件")

        # 读取图像数据
        image_array = dataset.ReadAsArray()

        return image_array

    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


if __name__ == '__main__':

    label_tuf_paths = r'data\yangben\test_labels'
    predict_tif_paths = r'data\yangben\test_predict'
    label_tuf_paths_li = [i for i in os.listdir(label_tuf_paths)
                          if i.split('.')[-1] == 'png']
    TIF_JDPJ_LI = []
    for TIF in label_tuf_paths_li:
        print('正在对%s进行精度评价...'%TIF)
        label_tif_path = label_tuf_paths + '\\' + TIF
        predict_tif_path = predict_tif_paths + '\\' + TIF.replace('.png', '_prediction.png')
        label = read_tiff_as_array(label_tif_path)
        predict = read_tiff_as_array(predict_tif_path)
        print(label.shape, predict.shape)
        y_true = label.reshape(-1)
        y_pred = predict.reshape(-1)
        y_pred = y_pred

        if y_true.shape[0] == y_pred.shape[0]:
            print('正在对%s进行精度评价...' % TIF)
            y_pred = y_pred.astype(int)
            if y_true.max() != 0:
                res = confusion_matrix(y_true, y_pred)
                print('混淆矩阵:', res)

                TN, FP, FN, TP = res.ravel()
                # TP, FN, FP, TN = res.ravel()
                print(TN, FP, FN, TP)

                mIOU = 0.5 * TP / (TP + FP + FN) + 0.5 * TN / (TN + FP + FN)
                # 将所有结果写进一个列表中
                TIF_JDPJ_LI.append([TIF, TN, FP, FN, TP, mIOU])

    DF = pd.DataFrame(TIF_JDPJ_LI, columns=['名称', 'TN', 'FP', 'FN', 'TP', 'mIOU'])
    # DF.to_excel('DF.xlsx', index=False)

    print('mIOU：', DF.mIOU.mean())
    print('accuracy:', (DF.TP.sum()+DF.TN.sum())/(DF.TP.sum()+DF.TN.sum()+DF.FP.sum()+DF.FN.sum()))

    print('运行完成！')
