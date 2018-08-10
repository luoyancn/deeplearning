# -*- coding:utf-8 -*-

from PIL import Image
import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('model_dir', '/root/tensorflow/model',
                     'The path of trained model saved. '
                     'Default is /root/tensorflow/model')
flags.DEFINE_string('image', '',
                     'The pictures to predict'
                     'Must be provided')
flags.DEFINE_string('ps', '',
                    'Params Server of TF cluster.Default is localhost:2223')
flags.DEFINE_integer('worker_index', 0,
                    'The index of TF Worker Servers.Default is 0')
flags.DEFINE_integer('ps_index', 0,
                    'The index of TF Params Servers.Default is 0')
FLAGS=flags.FLAGS

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

def restore_model(pic_array):
    # 重现计算图
    with tf.Graph().as_default() as gph:
        worker = FLAGS.worker_index
        ps = FLAGS.ps_index
        ps_device = '/job:ps/task:%s' % ps
        tower_device = '/job:worker/task:%s' % worker

        with tf.device(tower_device):
            # Create the model
            # 只需要对输入占位
            x = tf.placeholder(tf.float32, [None, 784])

        # Build the graph for the deep net
        y_conv, keep_prob, ps_vars, replica_vars, fetch_ops = deepnn(x, ps_device, tower_device)

        # y的最大值对应的索引号，就是预测的数字的值
        pre_value = tf.argmax(y_conv, 1)

        saver = tf.train.Saver()

        with tf.Session("grpc://%s" % FLAGS.ps) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 预测操作
                pre_value = sess.run(pre_value, feed_dict={x: pic_array, keep_prob: 1.0})
                return pre_value


def pre_dic(pic_path, wg_bg=False):
    # 读取图片
    img = Image.open(pic_path)
    # 用消除锯齿的方式，将图片resize 为28 × 28
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 将resize的图片转换为灰度图，并转换为矩阵的方式
    img_array = np.array(reIm.convert('L'))

    # mnist训练的图片要求黑底白字，因此训练之后的模型也只接收黑底白字的图片
    # 当推测的是白底黑字的单色通道图片时，需要对图片进行反色，变成黑底白字，只留下纯白和纯黑点
    # 如果推测的是黑底白字的图片，可以不用进行反色
    # 如果推测的是rgb彩色图片，则推测功能不可用
    # 该推测代码基于白底黑字的图片，所以需要进行反色。
    if not wg_bg:
        threshold = 50
        for i in range(28):
            for j in range(28):
                img_array[i][j] = 255 - img_array[i][j]
                if img_array[i][j] < threshold:
                    # 黑点
                    img_array[i][j] = 0
                else:
                    # 白点
                    img_array[i][j] = 255
    # 将图片整理为1 × 784的矩阵
    nm_array = img_array.reshape([1, 784])
    # 转换为浮点型
    nm_array = nm_array.astype(np.float32)
    # 将rbg从0-255变为1-255的数
    img_ready = np.multiply(nm_array, 1.0/255.0)
    return img_ready


def application(image_path, wg_bg=False):
    pic_array = pre_dic(image_path, wg_bg=wg_bg)
    pre_val = restore_model(pic_array)
    print(pre_val)

def main(unused_args):
    application(FLAGS.image)

if __name__ == '__main__':
    tf.app.run()