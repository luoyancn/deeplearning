# -*- coding:utf-8 -*-
import os

import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_TRAIN_PATH = '../../mnsit_tf_record/mnsit_train'
LABEL_TRAIN_PATH = '../../mnsit_tf_record/mnsit_train.txt'
TFRECORD_TRAIN = '../../mnsit_tf_record/mnsit_train_tfrecord'

IMAGE_TEST_PATH = '../../mnsit_tf_record/mnsit_test'
LABLE_TEST_PATH = '../../mnsit_tf_record/mnsit_test.txt'
TFRECORD_TEST = '../../mnsit_tf_record/mnsit_test_tfrecord'

DATA_PATH = '../../mnsit_tf_record/data'
RESIZE_HEIGHT = 28
RESIZE_WIDTH = 28


def write_tf_record(tf_record_name, image_path, label_path):
    # 生成tf record的writer
    writer = tf.python_io.TFRecordWriter(tf_record_name)
    num_pic = 0
    # 打开标签文件
    with open(label_path, 'r') as contents:
        for content in contents:
            val = content.split()
            img_path = image_path + '/' + val[0]
            img = Image.open(img_path)
            # 将图片转换为2进制数据
            img_raw = img.tobytes()
            labels = [0] * 10
            # label对应的标签位设置为1
            labels[int(val[1])] = 1

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    # 放入二进制图
                    'img_raw': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_raw])),
                    # 图片的label
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=labels))
            }))
            # 写入序列化之后的数据
            writer.write(example.SerializeToString())
            num_pic += 1
    writer.close()
    print('Write tf record successful')


def generate_tf_recorde():
    is_exist = os.path.exists(DATA_PATH)
    if not is_exist:
        os.makedirs(DATA_PATH)
    # 生成训练集
    write_tf_record(TFRECORD_TRAIN, IMAGE_TRAIN_PATH, LABEL_TRAIN_PATH)
    # 生成测试集
    write_tf_record(TFRECORD_TEST, IMAGE_TEST_PATH, LABLE_TEST_PATH)


def read_tf_record(tf_record_path):
    # 新建文件名队列
    file_name_queue = tf.train.string_input_producer([tf_record_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            # 标签第一个参数表示多少分类。由于mnsit是10个数字，因此是10种分类
            'label': tf.FixedLenFeature([10], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 将图片转换为784的矩阵
    img.set_shape([784])
    # 转换为0-1之间的浮点数。除以255是由于单色的最大值是255
    img = tf.cast(img, tf.float32) * (1./ 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tf_record(num, is_train=True):
    if is_train:
        tf_record_path = TFRECORD_TRAIN
    else:
        tf_record_path = TFRECORD_TEST
    img, label = read_tf_record(tf_record_path)
    # 使用2个线程进行数据提取，每次按照num的量进行数据的提取
    # 从总样本当中，顺序取出1000组数据，打乱顺序。每次输出
    # num组数据。如果某次取出的capacity少于min_after_dequeue，
    # 则从总样本当中提取重复数据，填满capacity
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=num, num_threads=2,
        capacity=1000, min_after_dequeue=700)
    # 图片和label则进行了随机化
    return img_batch, label_batch


if __name__ == '__main__':
    generate_tf_recorde()