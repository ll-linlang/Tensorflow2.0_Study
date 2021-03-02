#!/usr/bin/env python
#_*_coding:utf-8_*_
#author:linlang
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
import glob,os,random
import ResNet18_with_CBAM


#加载数据集路径
base_path = "./data"
#查看数据集长度 2295
img_list = glob.glob(os.path.join(base_path,"*/*.jpg"))
print(len(img_list))
# 对数据集进行分组
train_datagen = ImageDataGenerator(
        rescale=1./255,shear_range=0.1,zoom_range=0.1,
        width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,
        vertical_flip=True,validation_split=0.1)

test_data = ImageDataGenerator(rescale=1./255,validation_split=0.1)

train_generator = train_datagen.flow_from_directory(base_path,target_size=(300,300),
                                                   batch_size=16,
                                                   class_mode="categorical",
                                                   subset="training",seed=0)

validation_generator = test_data.flow_from_directory(base_path,target_size=(300,300),
                                                    batch_size=16,
                                                    class_mode="categorical",
                                                    subset="validation",seed=0
                                                    )

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)

model = ResNet18_with_CBAM.ResNet([2,2,2,2])
model.build(input_shape=(None,300,300,3))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, epochs=100, steps_per_epoch=2068//32,validation_data=validation_generator,
                    validation_steps=227//32)
