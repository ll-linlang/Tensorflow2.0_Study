#!/usr/bin/env python
#_*_coding:utf-8_*_
#author:linlang
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,AveragePooling2D,Dropout,Flatten,Dense,add,GlobalAveragePooling2D
from tensorflow.keras import Model,Sequential


class BasicBlock(Model):

    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.filters = filters
        self.strides = stride

        #第一小块
        self.c1 = Conv2D(filters, (3, 3), strides=stride, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        # 第二小块
        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()

        # 判断stride是否为1，若stride=1，说明卷积前后维度相同，可以直接相加；stride！=1，则进行同stride卷积操作，再进行相加
        if stride!=1:
            self.downsample =Sequential()
            self.downsample  = Conv2D(filters, (1, 1), strides=stride)
        else:
            self.downsample  = lambda x: x
        #
        self.a2 = Activation('relu')

    def call(self, inputs):

        # 1、先将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x)

        #2、然后将输入送入shortcut路径中
        identity = self.downsample(inputs)

        #3、最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        out = self.a2(y + identity)
        # out = self.a2(add([y, identity]) ) #两种加法均可

        return out

class ResNet18(Model):

    # 第一个参数 block_list：[2, 2, 2, 2] 4个Res Block，每个包含2个Basic Block
    # 第二个参数 num_classes：我们的全连接输出，取决于输出有多少类。
    def __init__(self, block_list, num_classes=10):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()

        # stem预处理层；实现起来比较灵活可以加 MAXPool2D，可以没有。
        self.stem = Sequential([Conv2D(64, (3, 3), strides=(1, 1)),
                                BatchNormalization(),
                                Activation('relu'),
                                MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])

        #创建4个ResBlock；注意第1项不一定以2倍形式扩张，都是比较随意的，这里都是经验值。
        self.layer1 = self.build_resblock(64, block_list[0])
        self.layer2 = self.build_resblock(128, block_list[1], stride=2)
        self.layer3 = self.build_resblock(256, block_list[2], stride=2)
        self.layer4 = self.build_resblock(512, block_list[3], stride=2)

        #使用GAP来替代该全连接层(即使用池化层的方式来降维), 以自适应的确定输出
        self.avgpool =GlobalAveragePooling2D()
        # 全连接层：为了分类
        self.fc = Dense(num_classes)

    def build_resblock(self, filter_num, blocks, stride=1):

        # may down sample 也许进行下采样,对于当前Res Block中的Basic Block，我们要求每个Res Block只有一次下采样的能力。
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))  # 这里stride设置为1，只会在第一个Basic Block做一个下采样。
        return res_blocks


    # 实现 Res Block； 创建一个Res Block
    def call(self, inputs, training=None):
        # __init__中准备工作完毕；下面完成前向运算过程。
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 做一个global average pooling，得到之后只会得到一个channel，不需要做reshape操作了。# shape为 [batchsize, channel]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)
        return x

