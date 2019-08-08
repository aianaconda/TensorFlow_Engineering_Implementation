# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./fashion/", one_hot=False)  #指定数据集路径

print ('输入数据:',mnist.train.images)
print ('输入数据的形状:',mnist.train.images.shape)
print ('输入数据的标签:',mnist.train.labels)

import pylab 
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()


#print ('输入数据打shape:',mnist.test.images.shape)
#print ('输入数据打shape:',mnist.validation.images.shape)
#print ('输入数据:',mnist.test.labels)
















