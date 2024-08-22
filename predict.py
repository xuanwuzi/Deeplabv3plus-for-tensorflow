# -*- coding: utf-8 -*-
import os
import numpy as np
from osgeo import gdal
from train import load_image
from nets.deeplab import Deeplabv3


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 模型路径
    model_path = r'Deeplabv3_512\Deeplabv3_512.h5'

    # 设置输入文件夹路径
    output_folder1 = r'data\yangben\test_images'
    # 设置输出文件夹路径
    output_folder2 = r'data\yangben\test_predict'
    # 输入图像的形状
    target_shape = (512, 512, 3)

    # 创建输出文件夹
    os.makedirs(output_folder2, exist_ok=True)

    print('正在预测...')
    # 模型加载
    # model = Deeplabv3(input_shape=target_shape, num_classes=2, backbone="mobilenet")
    model = Deeplabv3(input_shape=target_shape, num_classes=2, backbone="xception")

    model.load_weights(model_path)

    # 获取输入文件夹下的所有.tif文件路径
    input_files = []
    for file in os.listdir(output_folder1):
        if file.endswith('.png'):
            input_files.append(os.path.join(output_folder1, file))
    print('共有%d个影像...'%len(input_files))

    # 对每个输入图像进行预测
    for input_file in input_files:
        # print(input_file)
        # 加载图像
        image = load_image(input_file)
        # if image.max() != image.min():
        image = np.array([image])
        mask = image[0, :, :, 0].copy()
        mask[mask!=0] = 1
        #print(image.shape)
        # 模型预测
        predictions = model.predict(image)
        #print(predictions.shape)
        # predictions = np.argmax(predictions, axis=-1)[0,:,:]
        predictions = predictions[0, :, :, 0]
        yz = 0.5
        predictions[predictions >= yz] = 1
        predictions[predictions < yz] = 0
        predictions = predictions*mask
        predictions = predictions.astype(np.uint8)
        print(predictions.shape)

        #获取输出文件名
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder2, f"{filename.split('.')[0]}_prediction.png")
        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(output_file, predictions.shape[1],
                                       predictions.shape[0], 1, gdal.GDT_Byte)
        #获取原图像的GeoTransform，im_Proj，并给预测图象添加上
        dataset = gdal.Open(input_file)
        adf_GeoTransform = dataset.GetGeoTransform()
        im_Proj = dataset.GetProjection()
        output_dataset.SetGeoTransform(adf_GeoTransform)
        output_dataset.SetProjection(im_Proj)
        output_dataset.GetRasterBand(1).WriteArray(predictions)
        output_dataset.FlushCache()
        print(f"Predictions saved for {input_file} to {output_file}")

    print('DONE!')
