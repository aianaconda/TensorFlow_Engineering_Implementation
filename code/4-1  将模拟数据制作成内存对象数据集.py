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

#在内存中生成模拟数据
def GenerateData(batchsize = 100):
    train_X = np.linspace(-1, 1, batchsize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    yield train_X, train_Y       #以生成器的方式返回

#定义网络模型结构部分，这里只有占位符张量
Xinput = tf.placeholder("float",(None))  #定义两个占位符，用来接收参数
Yinput = tf.placeholder("float",(None))

#建立会话，获取并输出数据
training_epochs = 20  # 定义需要迭代的次数
with tf.Session() as sess:  # 建立会话（session）
    for epoch in range(training_epochs): #迭代数据集20遍
        for x, y in GenerateData(): #通过for循环打印所有的点
            xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput: x, Yinput: y})#通过静态图注入的方式，传入数据

            print(epoch,"| x.shape:",np.shape(xv),"| x[:3]:",xv[:3])
            print(epoch,"| y.shape:",np.shape(yv),"| y[:3]:",yv[:3])
     
    
#显示模拟数据点
train_data =list(GenerateData())[0]
plt.plot(train_data[0], train_data[1], 'ro', label='Original data')
plt.legend()
plt.show()