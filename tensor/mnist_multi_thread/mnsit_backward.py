# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnsit_forward
import mnist_generate_recode

# 每轮输入的文件数量
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVG_DECAY = 0.99
MODEL_SAVE_PATH = 'models'
MODEL_NAME = 'mnist_model'

TRAIN_NUM_EXAMPLES = 60000


def backward():
    x = tf.placeholder(tf.float32, [None, mnsit_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnsit_forward.OUTPUT_NODE])
    y = mnsit_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        TRAIN_NUM_EXAMPLES/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVG_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    img_batch, label_batch = mnist_generate_recode.get_tf_record(
        BATCH_SIZE, is_train=True)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            _, loss_value, learning_rate_val, step = sess.run(
                [train_op, loss, learning_rate, global_step],
                feed_dict={x: xs, y_: ys})
            if 0 == i % 1000:
                fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
                print(fmt.format(step, loss_value, learning_rate_val))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        coord.request_stop()
        coord.join(threads)

def main():
    backward()


if __name__ == '__main__':
    main()
