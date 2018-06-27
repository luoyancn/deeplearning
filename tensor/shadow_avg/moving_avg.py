# -*- coding:utf-8 -*-

import tensorflow as tf

# 定义变量，初始值为0.0。该代码的功能
# 在于不断更新w1参数，优化w1参数，滑动
# 平均做了一个w1的影子
w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)

# 设置滑动平均的衰减率为0.99
MOVING_AVG_DECAY = 0.99
# 实例化滑动平均类
ema = tf.train.ExponentialMovingAverage(
    MOVING_AVG_DECAY, global_step)

# 实际生产当中，自动将所有待训练的参数汇总为列表
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出初始的w1和w1滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    # 参数w1赋值为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新训练步骤和w1的值，模拟100轮迭代的结果
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))