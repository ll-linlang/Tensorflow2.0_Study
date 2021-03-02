#!/usr/bin/env python
#_*_coding:utf-8_*_
#author:linlang#!/usr/bin/env python
#_*_coding:utf-8_*_
#author:linlang
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,AveragePooling2D,Dropout,Flatten,Dense,add,GlobalAveragePooling2D
from tensorflow.keras import Model,Sequential
from cbam import *

def regurlarized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding="same",
                         use_bias=False,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(5e-4))
class BasicBlock(Model):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一小块
        self.conv1 = regurlarized_padded_conv(out_channels, kernel_size=3,
                                              strides=stride)
        self.bn1 = layers.BatchNormalization()

        # 第二小块
        self.conv2 = regurlarized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()
        ########注意力机制#################
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # 3.判断stride是否等于1，如果为1就是没有降采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regurlarized_padded_conv(self.expansion * out_channels,
                                                                 kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])

        else:
            self.shortcut = lambda x, _: x


    def call(self, inputs,training=False):

        # 1、先将输入通过卷积、BN层、激活层，计算F(x)
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)


        ########注意力机制###########
        out = self.ca(out) * out

        out = self.sa(out) * out

        # 2、然后将输入送入shortcut路径中，即F(x)+x或F(x)+Wx,再过激活函数
        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=6):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 预测理卷积
        self.stem = Sequential([
            regurlarized_padded_conv(64, kernel_size=3, strides=1),
            layers.BatchNormalization()
        ])
        # 创建4个残差网络
        self.layer1 = self.build_resblock(32, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(64, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256,layer_dims[2],stride=2)
        self.layer4 = self.build_resblock(512,layer_dims[3],stride=2)

        self.final_bn = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out,training=training)
        out = self.layer4(out,training=training)

        out = self.final_bn(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out

    #         self.final_bn = layers.BatchNormalization()
    #         self.avgpool =
    # 1.创建resBlock
    def build_resblock(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks
