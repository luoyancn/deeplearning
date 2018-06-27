# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data
import forward


STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = data.generate_dataset()
    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(
        tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(
        learning_rate).minimize(loss_total, global_step=global_step)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step,
                     feed_dict={x: X[start:end], y_: Y_[start:end]})
            if 0 ==  i % 2000:
                loss_val = sess.run(
                    loss_total, feed_dict={x:X, y_: Y_})
                learning_rate_val = sess.run(learning_rate)
                fmt = "After {:05d} steps, loss is {:.09f}, learing rate is {:.09f}"
                print(fmt.format(i, loss_val, learning_rate_val))
        # 在x和y轴，以步长为0.01,在-3到3之间生成二维坐标点
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        # 将xx和yy拉直，合并为2列矩阵，得到网格坐标点集合
        grid = np.c_[xx.ravel(), yy.ravel()]
        # 将网格坐标点输入神经网络，probs为输出
        probs = sess.run(y, feed_dict={x:grid})
        # probs的shape调整为xx的形式
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()

if __name__ == '__main__':
    backward()
