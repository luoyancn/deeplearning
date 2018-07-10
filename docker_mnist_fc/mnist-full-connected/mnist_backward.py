import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_forward

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
flags.DEFINE_float('moving_avg_decay', 0.99,
                   'The decay of moveing avg rate. '
                   'Default is 0.99')
flags.DEFINE_float('regularizer', 0.0001,
                   'The regularizer for Neural Network training. '
                   'Default is 0.0001')
flags.DEFINE_integer('steps', 50000,
                     'The total traning times. '
                     'Default is 50000')
flags.DEFINE_string('model_name', 'mnist_full_connected',
                    'The name of train model. '
                    'Default is mnist_full_connected')
flags.DEFINE_string('model_save_path', '/root/tensorflow/model',
                     'The models trained save path. '
                     'Default is mnist_models')
flags.DEFINE_string('training_data_path', '/root/mnist-full-connected/mnist_data',
                     'The path of training dataset path. '
                     'Default is mnist_data')
flags.DEFINE_string('log_dir', '/root/tensorflow/log',
                     'The path of trained logs path. '
                     'Default is mnist_fc_logs')
FLAGS = flags.FLAGS


def backward(mnist):

    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32, [None, mnist_forward.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_forward.OUTPUT_NODE], name='y_input')
        image_shaped_input = tf.reshape(
            x, [-1, mnist_forward.MNIST_SIZE, mnist_forward.MNIST_SIZE, 1])
        tf.summary.image(
            'input', image_shaped_input, mnist_forward.OUTPUT_NODE)

    y = mnist_forward.forward(x, FLAGS.regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate_base,
            global_step,
            mnist.train.num_examples/FLAGS.batch_size,
            FLAGS.learning_rate_decay,
            staircase=True, name='learning_rate'
        )
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train_step'):
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope('exponential_moving_average'):
        ema = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay, global_step)
        ema_op = ema.apply(tf.trainable_variables())

    with tf.name_scope('training'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        for i in range(FLAGS.steps):
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value, learning_rate_val, step, summary = sess.run(
                [train_op, loss, learning_rate, global_step, merged],
                feed_dict={x: batch_xs, y_: batch_ys})
            writer.add_summary(summary, i)

            if 0 == i % 1000:
                fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
                print(fmt.format(step, loss_value, learning_rate_val))
                saver.save(sess, os.path.join(
                    FLAGS.model_save_path, FLAGS.model_name),
                    global_step=global_step)
                accuracy_rate = sess.run(
                    accuracy, feed_dict={x: mnist.test.images,
                                         y_: mnist.test.labels})
                fmt = 'After {:d} steps, test accuracy rate is {:.09f}'
                print(fmt.format(step, accuracy_rate))

        writer.close()


def main(unused_args):
    mnist = input_data.read_data_sets(FLAGS.training_data_path, one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    tf.app.run()
