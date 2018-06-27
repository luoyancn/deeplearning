import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 30
seed = 2

rdm = np.random.RandomState(seed)
# 随机生成300 × 2的矩阵，表示300个坐标点
X = rdm.randn(300, 2)

# 构造正确答案，x*x + y*y<2为正确答案
Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]

# 对答案进行着色
Y_c = [['red' if y else 'blue'] for y in Y_]

# 矩阵进行转换，转换为n行2列，-1表示n行
X = np.vstack(X).reshape(-1, 2)
# 转换为n行1列
Y_ = np.vstack(Y_).reshape(-1, 1)

#print(X)
#print(Y_)
#print(Y_c)

# 画出数据集X各行第0列和第1列的点，即各行的（x0, x1）
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()

# 定义神经网络的输入，参数，输出以及前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 生成偏置项b
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b

# 输入数据占位
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 2行11列
w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 11行1列
w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])

# 输出层不进行激活
y = tf.matmul(y1, w2) + b2

# 定义损值函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
# 损值函数正则化
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义反向传播方法--无正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 40000
    for i in range(steps):
        start = (i * batch_size) % 300
        end = start + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
        if 0 == i % 2000:
            loss_mse_val = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print('After {:05d} steps, loss is {:.9f}'.format(i, loss_mse_val))
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)
    print(sess.run(w1))
    print(sess.run(b1))
    print(sess.run(w2))
    print(sess.run(b2))
    
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 定义反向传播方法--正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 40000
    for i in range(steps):
        start = (i * batch_size) % 300
        end = start + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
        if 0 == i % 2000:
            loss_mse_val = sess.run(loss_total, feed_dict={x:X, y_:Y_})
            print('After {:05d} steps, loss is {:.9f}'.format(i, loss_mse_val))

    # 在x和y轴，以步长为0.01,在-3到3之间生成二维坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    # 将xx和yy拉直，合并为2列矩阵，得到网格坐标点集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点输入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x:grid})
    # probs的shape调整为xx的形式
    probs = probs.reshape(xx.shape)
    print(sess.run(w1))
    print(sess.run(b1))
    print(sess.run(w2))
    print(sess.run(b2))
    
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()