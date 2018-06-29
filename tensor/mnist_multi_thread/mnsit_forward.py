# -*- coding:utf-8 -*-

import tensorflow as tf

# 28 × 28 个像素点，表示图片的像素值
INPUT_NODE = 784

# 表示输出10个数，表示0-9出现的概率
OUTPUT_NODE = 10

# 定义隐藏层个数
LAYER_NODE1 = 500


def forward(x, regularizer):

    def get_weight(shape, regularizer):
        # 随机生成参数，去掉偏离过大的正态分布
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        # 加入正则化
        if regularizer:
            tf.add_to_collection(
                'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    def get_bias(shape):
        b = tf.Variable(tf.zeros(shape))
        return b

    w1 = get_weight([INPUT_NODE, LAYER_NODE1], regularizer)
    b1 = get_bias([LAYER_NODE1])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER_NODE1, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y