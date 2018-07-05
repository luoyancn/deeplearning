# -*- coding:utf-8 -*-

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

RGB_CHANNEL = 3
VGG_PIX_SIZE = 224

# 正常显示中文标签,要求安装对应字体
mpl.rcParams['font.sans-serif']=['SimHei']
 # 正常显示正负号
mpl.rcParams['axes.unicode_minus']=False

def load_image(path, show=False):
    # 读入图片
    img = io.imread(path)
    # 将像素归一化到[0,1]之间
    img = img / 255.0

    # 获取图像最短边
    short_edge = min(img.shape[:2])
    # 将图像的长和宽都减去最短边的长度，然后求平均
    y = (img.shape[0] - short_edge) / 2
    x = (img.shape[1] - short_edge) / 2
    # 取出切分之后的中心图
    crop_img = img[int(y):int(y)+short_edge,
                   int(x):int(x)+short_edge]

    # 将图像resize为vgg6网络所识别的固定的224 × 224 分辨率
    re_img = transform.resize(crop_img, (VGG_PIX_SIZE, VGG_PIX_SIZE))
    # 将图片调整为[1, 224, 224, 3]维度的输出样式
    img_ready = re_img.reshape((1, VGG_PIX_SIZE, VGG_PIX_SIZE, RGB_CHANNEL))

    if show:
        fig = plt.figure('Centre and Resize')
        # 建立1行3列的子图，并将该图放到第一列
        ax0 = fig.add_subplot(131)
        # 添加子图标签
        ax0.set_xlabel('Original Picture')
        # 显示该图
        ax0.imshow(img)

        # 将该图放到第2列
        ax1 = fig.add_subplot(132)
        ax1.set_xlabel('Centre Picture')
        ax1.imshow(crop_img)

        # 将该图放到第3列
        ax2 = fig.add_subplot(133)
        ax2.set_xlabel('Resize Picture')
        ax2.imshow(re_img)

    return img_ready

def percent(value):
    return '%.2f%%' % (value * 100)
