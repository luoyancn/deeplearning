from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import argparse
import os
import sys
import time
  
from tensorflow.examples.tutorials.mnist import input_data
  
import tensorflow as tf
  
FLAGS = None
  

def deepnn(x, ps_device, tower_device):
    """deepnn builds the graph for a deep net for classifying digits.
  
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
  
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    ps_vars = {}
    replica_vars = []
    fetch_ops = []
  
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.device(tower_device):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
  
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1_ps, W_conv1 = weight_variable([5, 5, 1, 32], 'w-conv1', ps_device, tower_device)
    b_conv1_ps, b_conv1 = bias_variable([32], 'b-conv1', ps_device, tower_device)
    ps_vars.update({W_conv1_ps.name: W_conv1_ps,
                    b_conv1_ps.name: b_conv1_ps})
    replica_vars.extend([W_conv1,
                         b_conv1])
    with tf.device(tower_device):
        fetch_ops.append(W_conv1.assign(W_conv1_ps.read_value()))
        fetch_ops.append(b_conv1.assign(b_conv1_ps.read_value()))
  
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  
        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)
  
    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2_ps, W_conv2 = weight_variable([5, 5, 32, 64], 'w-conv2', ps_device, tower_device)
    b_conv2_ps, b_conv2 = bias_variable([64], 'b-conv2', ps_device, tower_device)
    ps_vars.update({W_conv2_ps.name: W_conv2_ps,
                    b_conv2_ps.name: b_conv2_ps})
    replica_vars.extend([W_conv2,
                         b_conv2])
    with tf.device(tower_device):
        fetch_ops.append(W_conv2.assign(W_conv2_ps.read_value()))
        fetch_ops.append(b_conv2.assign(b_conv2_ps.read_value()))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  
        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)
  
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1_ps, W_fc1 = weight_variable([7 * 7 * 64, 1024], 'w-fc1', ps_device, tower_device)
    b_fc1_ps, b_fc1 = bias_variable([1024], 'b-fc1', ps_device, tower_device)
    ps_vars.update({W_fc1_ps.name: W_fc1_ps,
                    b_fc1_ps.name: b_fc1_ps})
    replica_vars.extend([W_fc1,
                         b_fc1])
    with tf.device(tower_device):
        fetch_ops.append(W_fc1.assign(W_fc1_ps.read_value()))
        fetch_ops.append(b_fc1.assign(b_fc1_ps.read_value()))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.device(tower_device):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
    # Map the 1024 features to 10 classes, one for each digit
    W_fc2_ps, W_fc2 = weight_variable([1024, 10], 'w-fc2', ps_device, tower_device)
    b_fc2_ps, b_fc2 = bias_variable([10], 'b-fc2', ps_device, tower_device)
    ps_vars.update({W_fc2_ps.name: W_fc2_ps,
                    b_fc2_ps.name: b_fc2_ps})
    replica_vars.extend([W_fc2,
                         b_fc2])
    with tf.device(tower_device):
        fetch_ops.append(W_fc2.assign(W_fc2_ps.read_value()))
        fetch_ops.append(b_fc2.assign(b_fc2_ps.read_value()))
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
    return y_conv, keep_prob, ps_vars, replica_vars, fetch_ops
  
  
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
  
def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
  
  
def weight_variable(shape, name, ps_device, tower_device):
    """weight_variable generates a weight variable of a given shape."""
    with tf.device(ps_device):
        initial = tf.truncated_normal(shape, stddev=0.1)
        ps_var = tf.Variable(initial, name='ps-'+name)
    with tf.device(tower_device):
        replica_var = tf.Variable(initial, name=name)
    return ps_var, replica_var
  
  
def bias_variable(shape, name, ps_device, tower_device):
    """bias_variable generates a bias variable of a given shape."""
    with tf.device(ps_device):
        initial = tf.constant(0.1, shape=shape)
        ps_var = tf.Variable(initial, name='ps-'+name)
    with tf.device(tower_device):
        replica_var = tf.Variable(initial, name=name)
    return ps_var, replica_var
  
  
def main(argv):
    # Import data
    try:
        os.makedirs(FLAGS.model_save_path)
    except OSError as exc:
        pass
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    worker = FLAGS.worker_index
    ps = FLAGS.ps_index
    ps_device = '/job:ps/task:%s' % ps
    tower_device = '/job:worker/task:%s' % worker
  
    with tf.device(tower_device):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
  
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])
  
    # Build the graph for the deep net
    y_conv, keep_prob, ps_vars, replica_vars, fetch_ops = deepnn(x, ps_device, tower_device)
    global_step = tf.Variable(0, trainable=False)
    with tf.device(tower_device):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        optimizer = tf.train.AdamOptimizer(1e-4, name='apply_gradients')
        gradients = optimizer.compute_gradients(cross_entropy, var_list=replica_vars)
        ps_gradients = [(grad, ps_vars['ps-'+replica_var.name]) for grad, replica_var in gradients]
        train_step = optimizer.apply_gradients(ps_gradients, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
    for tvariable in tf.trainable_variables():
        print(tvariable.name + ": " + tvariable.device)

    saver = tf.train.Saver()

    ps_url = FLAGS.ps
    cluster_spec = os.environ.get('CLUSTER_SPEC', '')
    ps_and_worker = cluster_spec.split(',')
    param_servers = [p for p in ps_and_worker if p.startswith('ps')]
    if param_servers:
        try:
            ps_urls = param_servers[0].split('|')[1:]
            ps_url = ps_urls[0]
        except Exception:
            pass

    with tf.Session("grpc://%s" % ps_url) as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        graph_writer = tf.summary.FileWriter('tf_summary', sess.graph)
        graph_writer.close()
        last_batch_time = time.time()
        for i in range(int(1000/2)):
            for w in range(2):
                next_batch = mnist.train.next_batch(50)
                if (w + 1) % (worker + 1) == 0:
                    batch = next_batch

            if i % 100 == 0:

                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                print('batch time %s' % (time.time() - last_batch_time, ))
                last_batch_time = time.time()
            tf.group(*fetch_ops).run()
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            saver.save(sess, os.path.join(FLAGS.model_save_path, FLAGS.model_name), global_step=global_step)
  
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/root/data',
                        help='Directory for storing input data')
    parser.add_argument('--model_save_path', type=str,
                        default='/root/models',
                        help='Directory for storing training data')
    parser.add_argument('--model_name', type=str,
                        default='mnist',
                        help='Name of models')
    parser.add_argument('--ps', type=str,
                        default='localhost:2223',
                        help='Params Server of TF cluster.Default is localhost:2223')
    parser.add_argument('--worker_index', type=int,
                        default=0,
                        help='The index of TF Worker Servers.Default is 0')
    parser.add_argument('--ps_index', type=int,
                        default=0,
                        help='The index of TF Params Servers.Default is 0')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
