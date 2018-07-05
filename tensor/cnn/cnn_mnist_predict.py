#coding:utf-8
from PIL import Image, ImageFilter
import tensorflow as tf
import cnn_model
#import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string("model_dir",
                    "/root/tensorflow/model",
                    "The directory for saving training chekcpoint and graph")
flags.DEFINE_string("image",
                    None,
                    "The image to predict")

FLAGS = flags.FLAGS

def imageprepare(image):
    im = Image.open(image)

    im = im.convert('L')
    if im.size[0] != 28 or im.size[1] != 28:
        im = im.resize((28, 28))
    tv = list(im.getdata())
    tva = [(255-x)*1.0/255.0 for x in tv]
    return tva

result = imageprepare(FLAGS.image)

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, [None, 784])

y_conv = cnn_model.cnn_model(x, keep_prob)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.model_dir + "/model.ckpt")

    prediction = tf.argmax(y_conv,1)
    predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)

    print('Predict Result:' + str(predint[0]))
