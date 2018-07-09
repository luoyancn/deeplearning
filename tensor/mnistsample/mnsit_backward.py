# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnsit_forward

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 200,
                     'The number of pictures feed to Neural Network once. '
                     'Should be > 0 ,default is 200')
flags.DEFINE_float('learning_rate_base', 0.8,
                   'The base learning rate for Neural Network training '
                   'Default is 0.8')
flags.DEFINE_float('learning_rate_decay', 0.99,
                   'The decay of learning rate. '
                   'Default is 0.99')
flags.DEFINE_float('regularizer', 0.0001,
                   'The regularizer for Neural Network training. '
                   'Default is 0.0001')
flags.DEFINE_float('moving_avg_decay', 0.99,
                   'The moving avg decay for traning. '
                   'Default is 0.99')
flags.DEFINE_integer('steps', 50000,
                     'The total traning times. '
                     'Default is 50000')
flags.DEFINE_string('model_name', 'mnist_full_connected',
                    'The name of train model. '
                    'Default is mnist_full_connected')
flags.DEFINE_string('model_save_path', 'mnist_models',
                     'The total traning times. '
                     'Default is 50000')
flags.DEFINE_string('training_data_path', 'mnist_data',
                     'The total traning times. '
                     'Default is 50000')
FLAGS = flags.FLAGS


def backward(mnsit):
    x = tf.placeholder(tf.float32, [None, mnsit_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnsit_forward.OUTPUT_NODE])
    y = mnsit_forward.forward(x, FLAGS.regularizer)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate_base,
        global_step,
        mnsit.train.num_examples/FLAGS.batch_size,
        FLAGS.learning_rate_decay,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(
        FLAGS.moving_avg_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(FLAGS.steps):
            xs, ys = mnsit.train.next_batch(FLAGS.batch_size)
            feed_dict={x: xs, y_: ys}
            _, loss_value, learning_rate_val, step = sess.run(
                [train_op, loss, learning_rate, global_step],
                feed_dict=feed_dict)
            if 0 == i % 100:
                fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
                print(fmt.format(step, loss_value, learning_rate_val))

                saver.save(sess, os.path.join(
                    FLAGS.model_save_path, FLAGS.model_name),
                    global_step=global_step)
                accuracy_rate = sess.run(
                    accuracy, feed_dict={x: mnsit.test.images,
                                         y_: mnsit.test.labels})
                fmt = 'After {:d} steps, test accuracy rate is {:.09f}'
                print(fmt.format(step, accuracy_rate))


def main(unused_args):
    mnsit = input_data.read_data_sets(FLAGS.training_data_path, one_hot=True)
    backward(mnsit)


if __name__ == '__main__':
    tf.app.run()
