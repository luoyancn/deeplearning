# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnsit_forward_cnn_lenet5

# 每轮输入的文件数量
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVG_DECAY = 0.99
MODEL_SAVE_PATH = 'models'
MODEL_NAME = 'mnist_model'


def backward(mnsit):
    # 卷积的输入要求必须是4维张量
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE, 
        mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
        mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
        mnsit_forward_cnn_lenet5.MNIST_IMAGE_CHANNELS])
    y_ = tf.placeholder(tf.float32, [
        None, mnsit_forward_cnn_lenet5.OUTPUT_NODE])

    y = mnsit_forward_cnn_lenet5.forward(
        x, train=True, regularizer=REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnsit.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVG_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnsit.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
                mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
                mnsit_forward_cnn_lenet5.MNIST_IMAGE_CHANNELS
            ))
            _, loss_value, learning_rate_val, step = sess.run(
                [train_op, loss, learning_rate, global_step],
                feed_dict={x: reshaped_xs, y_: ys})
            if 0 == i % 1000:
                fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
                print(fmt.format(step, loss_value, learning_rate_val))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('data', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
