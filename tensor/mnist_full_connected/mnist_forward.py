# -*- coding:utf-8 -*-

import tensorflow as tf

# 28 × 28 个像素点，表示图片的像素值
INPUT_NODE = 784

# mnist数据集的像素
MNIST_SIZE = 28

# 表示输出10个数，表示0-9出现的概率
OUTPUT_NODE = 10

# 定义隐藏层个数
LAYER_NODE1 = 500

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)  ##直方图


def forward(x, regularizer=None):
    def get_weight(shape, regularizer, name):
        # 随机生成参数，去掉偏离过大的正态分布
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1, name=name))
        # 加入正则化
        if regularizer:
            tf.add_to_collection(
                'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    def get_bias(shape, name):
        b = tf.Variable(tf.zeros(shape))
        return b

    with tf.name_scope('layer'):
        with tf.name_scope('Input_layer'):
            with tf.name_scope('w1'):
                w1 = get_weight([INPUT_NODE, LAYER_NODE1], regularizer, name='w1')
                variable_summaries(w1)
            with tf.name_scope('b1'):
                b1 = get_bias([LAYER_NODE1], name='b1')
                variable_summaries(b1)
            with tf.name_scope('L1'):
                L1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='L1')
        with tf.name_scope('Output_layer'):
            with tf.name_scope('w2'):
                w2 = get_weight([LAYER_NODE1, OUTPUT_NODE], regularizer, name='w2')
                variable_summaries(w2)
            with tf.name_scope('b2'):
                b2 = get_bias([OUTPUT_NODE], name='b2')
                variable_summaries(b2)
        y = tf.matmul(L1, w2) + b2
    return y