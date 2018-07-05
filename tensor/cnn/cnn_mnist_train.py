#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import cnn_model


flags = tf.app.flags

flags.DEFINE_string("data_dir",
                    "/root/mnist/data",
                    "The directory for training and test dataset")

flags.DEFINE_string("model_dir",
                    "/root/tensorflow/model",
                    "The directory for saving training chekcpoint and graph")

flags.DEFINE_string("log_dir",
                    "/root/tensorflow/log",
                    "The directory for saving training chekcpoint and graph")

flags.DEFINE_integer("iterations",
                     10000,
                     "the steps of training")

flags.DEFINE_integer("batch_size",
                     100,
                     "The batch size of training")

flags.DEFINE_float("learning_rate",
                   1e-4,
                   "The learning_rate of training")

flags.DEFINE_integer("display_step",
                     100,
                     "The display step of training")

FLAGS = flags.FLAGS

#begin time
start = time.clock()
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

"""
x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
"""
keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32,[None, 784])
y_conv = cnn_model.cnn_model(x, keep_prob)

"""
"""
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

saver = tf.train.Saver()
sess = tf.Session()
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

with tf.name_scope('summary'):
    with tf.name_scope('accuracy'):
        tf.summary.scalar('accuracy', accuracy)

my_summary_op = tf.summary.merge_all()
my_file_writer = tf.summary.FileWriter(FLAGS.log_dir)

for i in range(FLAGS.iterations):
    batch = mnist.train.next_batch(FLAGS.batch_size)
    if i % FLAGS.display_step == 0:
        train_accuracy = accuracy.eval(session = sess,
                                       feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, train_accuracy %g" %(i, train_accuracy))
    train_step.run(session = sess, feed_dict = {x:batch[0], y_:batch[1],
                   keep_prob:0.5})

    if i % 5 == 0:
        summary = sess.run(my_summary_op, feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})
        my_file_writer.add_summary(summary, 1)

print("test accuracy %g" %accuracy.eval(session = sess,
      feed_dict = {x:mnist.test.images, y_:mnist.test.labels,
                   keep_prob:1.0}))
saver.save(sess, FLAGS.model_dir + "/model.ckpt")

end = time.clock()
print("running time is %g s") % (end-start)
