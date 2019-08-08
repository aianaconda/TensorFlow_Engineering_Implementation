# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#在内存中生成模拟数据
def GenerateData(training_epochs ,batchsize = 100):
    for i in range(training_epochs):
        train_X = np.linspace(-1, 1, batchsize)   #train_X为-1到1之间连续的100个浮点数
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
        
        yield shuffle(train_X, train_Y),i

Xinput = tf.placeholder("float",(None))  #定义两个占位符，用来接收参数
Yinput = tf.placeholder("float",(None))


training_epochs = 20  # 定义需要迭代的次数

with tf.Session() as sess:  # 建立会话（session）

    for (x, y) ,ii in GenerateData(training_epochs):
        xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput: x, Yinput: y})#通过静态图注入的方式，传入数据
        print(ii,"| x.shape:",np.shape(xv),"| x[:3]:",xv[:3])
        print(ii,"| y.shape:",np.shape(yv),"| y[:3]:",yv[:3])

     
    
#显示模拟数据点
train_data =list(GenerateData(1))[0]
plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
plt.legend()
plt.show()