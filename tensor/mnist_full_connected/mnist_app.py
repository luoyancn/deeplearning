# -*- coding:utf-8 -*-

from PIL import Image
import tensorflow as tf
import numpy as np

import mnist_backward
import mnist_forward

flags = tf.app.flags
flags.DEFINE_string('model_dir', '/root/tensorflow/model',
                     'The path of trained model saved. '
                     'Default is /root/tensorflow/model')
flags.DEFINE_string('image', '',
                     'The pictures to predict'
                     'Must be provided')
FLAGS=flags.FLAGS


def restore_model(pic_array):
    # 重现计算图
    with tf.Graph().as_default() as gph:
        # 只需要对输入占位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)

        # y的最大值对应的索引号，就是预测的数字的值
        pre_value = tf.argmax(y, 1)

        variable_avg = tf.train.ExponentialMovingAverage(mnist_backward.FLAGS.moving_avg_decay)
        variable_to_restore = variable_avg.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 预测操作
                pre_value = sess.run(pre_value, feed_dict={x: pic_array})
                return pre_value


def pre_dic(pic_path, wg_bg=False):
    # 读取图片
    img = Image.open(pic_path)
    # 用消除锯齿的方式，将图片resize 为28 × 28
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 将resize的图片转换为灰度图，并转换为矩阵的方式
    img_array = np.array(reIm.convert('L'))

    # mnist训练的图片要求黑底白字，因此训练之后的模型也只接收黑底白字的图片
    # 当推测的是白底黑字的单色通道图片时，需要对图片进行反色，变成黑底白字，只留下纯白和纯黑点
    # 如果推测的是黑底白字的图片，可以不用进行反色
    # 如果推测的是rgb彩色图片，则推测功能不可用
    # 该推测代码基于白底黑字的图片，所以需要进行反色。
    if not wg_bg:
        threshold = 50
        for i in range(28):
            for j in range(28):
                img_array[i][j] = 255 - img_array[i][j]
                if img_array[i][j] < threshold:
                    # 黑点
                    img_array[i][j] = 0
                else:
                    # 白点
                    img_array[i][j] = 255
    # 将图片整理为1 × 784的矩阵
    nm_array = img_array.reshape([1, 784])
    # 转换为浮点型
    nm_array = nm_array.astype(np.float32)
    # 将rbg从0-255变为1-255的数
    img_ready = np.multiply(nm_array, 1.0/255.0)
    return img_ready


def application(image_path, wg_bg=False):
    pic_array = pre_dic(image_path, wg_bg=wg_bg)
    pre_val = restore_model(pic_array)
    print(pre_val)

def main(unused_args):
    application(FLAGS.image)

if __name__ == '__main__':
    tf.app.run()

'''
if __name__ == '__main__':
    # 黑底白字的图片推测
    application('../../mnsit_pic/blackbg_whitefg/111.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/111.png', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1110.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1111.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1118.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1119.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1112.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1113.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1114.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1115.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1116.jpg', wg_bg=True)
    application('../../mnsit_pic/blackbg_whitefg/1117.jpg', wg_bg=True)
    # 白底黑字的图片推测
    application('../../mnsit_pic/whitebg_blackfg/1.png')
    application('../../mnsit_pic/whitebg_blackfg/2.png')
    application('../../mnsit_pic/whitebg_blackfg/3.png')
    application('../../mnsit_pic/whitebg_blackfg/4.png')
    application('../../mnsit_pic/whitebg_blackfg/5.png')
    application('../../mnsit_pic/whitebg_blackfg/6.png')
    application('../../mnsit_pic/whitebg_blackfg/7.png')
    application('../../mnsit_pic/whitebg_blackfg/8.png')
    application('../../mnsit_pic/whitebg_blackfg/9.png')
'''
