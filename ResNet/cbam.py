#!/usr/bin/env python
#_*_coding:utf-8_*_
#author:linlang
import tensorflow as tf
from tensorflow.keras import layers,Sequential,regularizers,optimizers
import tensorflow.keras as keras

#定义一个3 * 3 的卷积，kernel_initializer=“he_normal”,"logrot_normal"

def regurlarized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding="same",
                         use_bias=False,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(5e-4))

#通道注意力机制
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()

        self.conv1 = layers.Conv2D(in_planes // ration, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)

        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4),
                                   use_bias=True)

        def call(self, inputs):
            avg = self.avg(inputs)
            max = self.max(inputs)
            avg = layers.Reshape((1, 1, avg.shape[1]))(avg)

            max = layers.Reshape((1, 1, max.shape[1]))(max)

            avg_out = self.conv2(self.conv1(avg))
            max_out = self.conv2(self.conv1(max))

            out = avg_out + max_out
            out = tf.nn.sigmoid(out)
            return out

#空间注意力机制
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regurlarized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)

        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)

        return out

