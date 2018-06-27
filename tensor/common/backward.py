# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data
import forward


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y, Y_c = data.generate_dataset()
    y = forward.forward()