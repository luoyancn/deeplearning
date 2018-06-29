# -*- coding:utf-8 -*-

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnsit_forward
import mnsit_backward
import mnist_generate_recode

TEST_INTERVAL_SECS = 5
TEST_NUM = 10000


def check_accuracy():
    with tf.Graph().as_default() as grh:
        x = tf.placeholder(tf.float32, [None, mnsit_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnsit_forward.OUTPUT_NODE])
        y = mnsit_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(
            mnsit_backward.MOVING_AVG_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        img_batch, label_batch = mnist_generate_recode.get_tf_record(TEST_NUM, is_train=False)

        while 1:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnsit_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    xs, ys = sess.run([img_batch, label_batch])

                    accuracy_rate = sess.run(
                        accuracy, feed_dict={x: xs, y_: ys})
                    fmt = 'After {:s} steps, test accuracy rate is {:.09f}'
                    print(fmt.format(global_step, accuracy_rate))

                    coord.request_stop()
                    coord.join(threads)
            time.sleep(TEST_INTERVAL_SECS)


def main():
    check_accuracy()


if __name__ == '__main__':
    main()
