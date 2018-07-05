#coding:utf-8
import tensorflow as tf

"""
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

"""
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize()        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides()   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def cnn_model(x, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # x_image -> [batch, in_height, in_width, in_channels]
    #            [batch, 28, 28, 1]
    # W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
    #            [5, 5, 1, 32]
    # output  -> [batch, out_height, out_width, out_channels]
    #            [batch, 28, 28, 32]
    h_pool1 = max_pool_2x2(h_conv1)
    # h_conv1 -> [batch, in_height, in_weight, in_channels]
    #            [batch, 28, 28, 32]
    # output  -> [batch, out_height, out_weight, out_channels]
    #            [batch, 14, 14, 32]

    """
    h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 64)
    """
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool1 -> [batch, 14, 14, 32]
    # W_conv2 -> [5, 5, 32, 64]
    # output  -> [batch, 14, 14, 64]
    h_pool2 = max_pool_2x2(h_conv2)
    # h_conv2 -> [batch, 14, 14, 64]
    # output  -> [batch, 7, 7, 64]

    """
    h_pool2(batch, 7, 7, 64) -> h_fc1(1, 1024)
    """
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    """
    Dropout

    h_fc1 -> h_fc1_drop,
    """
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """
    """
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv
