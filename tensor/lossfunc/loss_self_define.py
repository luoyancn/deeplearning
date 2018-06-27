# coding:utf-8
import tensorflow as tf
import numpy as np


batch_size = 8
seed = 23456
COST = 1
PROFIT = 9

rdm = np.random.RandomState(seed)
# 生成32 ×2 的矩阵数据集，数据范围为0-1之间的数据x1和x2
X = rdm.rand(32, 2)

# 对x1和x2求和，并加上随机噪声，构建模拟的标准答案
Y_ = [[x1 + x2 + (rdm.rand()/10.0 - 0.05)] for (x1, x2) in X]

# 定义神经网络输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损值函数，使用自定义的损值函数
loss_self_define = tf.reduce_sum(
    tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ -y) *PROFIT))
#loss_self_define = tf.reduce_mean(
#    tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ -y) *PROFIT))
# 反向传播方法是用梯度下降
train_step = tf.train.GradientDescentOptimizer(
    0.001).minimize(loss_self_define)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 20000
    for i in range(steps):
        start = (i * batch_size) %32
        end = (i * batch_size) % 32 + batch_size
        result = sess.run(
            train_step, feed_dict={x:X[start:end], y_: Y_[start:end]})
        if 0 == i % 500:
            print('After {:04d} traning steps'.format(i), sess.run(w1))
            print(sess.run(loss_self_define, feed_dict={x: X, y_:Y_}))
    print(sess.run(w1))
