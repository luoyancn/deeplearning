# coding:utf-8
import tensorflow as tf
 
# 最初的学习率
LEARNING_RATE_BASE = 0.1
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 学习率更新频度，一般设定为样本总数/batch_size
LEARNING_RATE_STEP = 1
 
# 只进行计数，不参与训练
global_step = tf.Variable(0, trainable=False)
 
# 定义指数衰减学习率，学习率不再是固定的，而是不断变化的
learning_rate_dynamic = tf.train.exponential_decay(
    LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP,
    LEARNING_RATE_DECAY, staircase=True
)
 
w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(
    learning_rate_dynamic).minimize(loss, global_step=global_step)
 
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(100):
        _,learning_rate_val,w_val,loss_val = sess.run(
            [train_step, learning_rate_dynamic, w, loss])
        print('After %s steps: w is %f, loss is %f, and learning rate is %f' % (
            i, w_val, loss_val, learning_rate_val))
