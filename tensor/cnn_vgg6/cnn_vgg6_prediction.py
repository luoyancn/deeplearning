# -*- coding:utf-8 -*-

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cnn_vgg6
import cnn_vgg6_utils
from vgg6_labels import labels

fig = plt.figure('Top-5 prediction:')

def prediction():
    parser = argparse.ArgumentParser(prog='predict',
        description='Predicte the image.')
    parser.add_argument('-i', '--image-path', dest='image_path',
        type=str, help='The path of image path')
    parser.add_argument('-s', '--show', dest='show',
        default=False, action="store_true",
        help='Show the predict result with picture')

    args = parser.parse_args()
    image_path = args.image_path
    if not os.path.exists(image_path):
        print('Please ensure the image file %s exsit' % image_path)
        return

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [1, cnn_vgg6_utils.VGG_PIX_SIZE,
                                cnn_vgg6_utils.VGG_PIX_SIZE,
                                cnn_vgg6_utils.RGB_CHANNEL])

        # 读取保存在npy文件当中的模型参数
        vgg = cnn_vgg6.Vgg16()
        # 复现原有模型
        vgg.forward(images)

        # 使用load_image处理输入图片
        # 由于vgg6模型处理的是224 × 224的rgb 3通道图片
        # 推断操作时，每次输入一张图片，因此，
        # load_image处理之后的图片应该为
        # [1, 224, 224, 3]的数据样式
        probability = sess.run(
            vgg.prob, feed_dict={
                images: cnn_vgg6_utils.load_image(
                    args.image_path, show=args.show)})

        # 获取推测数据排序后的最高五个结果（索引值）
        # 这些索引值就是vgg6_labels.py文件的键
        top5 = np.argsort(probability[0])[-1:-6:-1]
        print("top5:",top5)
        values = []
        bar_label = []
        for n, i in enumerate(top5):
            print("n:",n)
            print("i:",i)
            values.append(probability[0][i])
            bar_label.append(labels[i])
            print(i, ":", labels[i], "----",
                  cnn_vgg6_utils.percent(probability[0][i]))
        if args.show:
            # 构建1行1列的子图
            ax = fig.add_subplot(111)
            # 构建柱状图，下标，高度，柱子的label，柱子的宽度和颜色
            ax.bar(range(len(values)), values,
                   tick_label=bar_label, width=0.5, fc='g')
            # 设置y轴的标签
            ax.set_ylabel(u'probabilityit')
            # 设置x轴的标签
            ax.set_title(u'Top-5')
            for a,b in zip(range(len(values)), values):
                ax.text(a, b+0.0005, cnn_vgg6_utils.percent(b),
                        ha='center', va = 'bottom', fontsize=7)
            plt.show()


if __name__ == '__main__':
    prediction()