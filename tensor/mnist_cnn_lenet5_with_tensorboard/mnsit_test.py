# -*- coding:utf-8 -*-

import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnsit_forward_cnn_lenet5
import mnsit_backward

TEST_INTERVAL_SECS = 5


def check_accuracy(mnist):
    with tf.Graph().as_default() as grh:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples, 
            mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
            mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
            mnsit_forward_cnn_lenet5.MNIST_IMAGE_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnsit_forward_cnn_lenet5.OUTPUT_NODE])
        y = mnsit_forward_cnn_lenet5.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(
            mnsit_backward.MOVING_AVG_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while 1:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnsit_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]
                    reshaped_xs = np.reshape(mnist.test.images, (
                        mnist.test.num_examples,
                        mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
                        mnsit_forward_cnn_lenet5.MNIST_IMAGE_SIZE,
                        mnsit_forward_cnn_lenet5.MNIST_IMAGE_CHANNELS
                    ))
                    accuracy_rate = sess.run(
                        accuracy, feed_dict={x: reshaped_xs,
                                             y_: mnist.test.labels})
                    fmt = 'After {:s} steps, test accuracy rate is {:.09f}'
                    print(fmt.format(global_step, accuracy_rate))
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('data', one_hot= True)
    check_accuracy(mnist)


if __name__ == '__main__':
    main()