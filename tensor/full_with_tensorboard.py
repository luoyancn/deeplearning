# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def forward(x, regularizer):
    def get_weight(shape, regularizer, name):
        # 随机生成参数，去掉偏离过大的正态分布
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        # 加入正则化
        if regularizer:
            tf.add_to_collection(
            'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w
    # 设定偏置项
    def get_bias(shape, name):
        b = tf.Variable(tf.zeros(shape))
        return b

    with tf.name_scope('layer'):
        with tf.name_scope('Input_layer'):
            with tf.name_scope('w1'):
                w1 = get_weight([784, 500], regularizer, name='w1')
                variable_summaries(w1)
            with tf.name_scope('b1'):
                b1 = get_bias([500], name='b1')
                variable_summaries(b1)
            with tf.name_scope('L1'):
                L1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='L1')
        with tf.name_scope('Output_layer'):
            with tf.name_scope('w2'):
                w2 = get_weight([500, 10], regularizer, name='w2')
                variable_summaries(w2)
            with tf.name_scope('b2'):
                b2 = get_bias([10], name='b2')
                variable_summaries(b2)
            y = tf.matmul(L1, w2) + b2
    return y

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x_input')#表示图片的像素值
    y_ = tf.placeholder(tf.float32, [None, 10],name='y_input')#表示图片的标签，即1个包含10个元素的数组
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])#由于tf.summary.image函数接受的是4阶张量，因此需要转换一次
    tf.summary.image('input', image_shaped_input, 10)

y = forward(x, 0.001)#前向传播网络的正则化系数设置为0.001

with tf.name_scope('loss'):
    # 构建交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 构建基于交叉熵的损值函数
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 由于前向传播网络使用了正则化，同样的，需要对损值函数进行正则化
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)

mnist = input_data.read_data_sets('data', one_hot=True)#载入mnist数据集
global_step = tf.Variable(0, trainable=False)

with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(
        0.8, global_step,mnist.train.num_examples/200,
        0.99, staircase=True
    )
    tf.summary.scalar('learning_rate', learning_rate)

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.name_scope('exponential_moving_average'):
    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

with tf.name_scope('training'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state('model_save_path')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    writer = tf.summary.FileWriter('logdir', sess.graph)
    for i in range(10000):
        xs, ys = mnist.train.next_batch(200)#每次喂入200张图片数据
        feed_dict={x: xs, y_: ys}
        _, loss_value, learning_rate_val, step, summary = sess.run([train_op, loss, learning_rate, global_step, merged],feed_dict=feed_dict)
        writer.add_summary(summary, i)
        if 0 == i % 100:
            saver.save(sess, os.path.join('model_save_path', 'model_name'), global_step=global_step)
            fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
            print(fmt.format(step, loss_value, learning_rate_val))
            #使用测试集计算正确率
            accuracy_rate = sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})
            fmt = 'After {:05d} steps, test accuracy rate is {:.09f}'
            print(fmt.format(step, accuracy_rate))
    writer.close()