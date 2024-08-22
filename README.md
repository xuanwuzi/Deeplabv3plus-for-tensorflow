# Deeplabv3plus-for-tensorflow
The project is developed using TensorFlow 2.0.

本项目基于python3.8环境, 必要的库tensoflow=2.10等。

目的：实现图像语义分割2分类

用法：
1.准备好样本数据集后，运行data_progress.py进行样本数据集随机划分为训练集、验证集、测试集

2.运行train.py进行模型的训练

3.运行predict.py进行在测试集上预测（也可针对新的数据进行预测）

4.运行calculate.py进行模型的精度评估
