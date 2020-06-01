# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:28:14 2020
@author: admin
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def GenerateData(training_epochs,batchsize=100):
    for i in range(training_epochs):
        train_X=np.linspace(-1,1,batchsize)
        train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3
        yield shuffle(train_X,train_Y),i
    #yield对象以生成器的方式返回对象，该对象只使用一次，之后便会自动销毁，可以为系统节省大量的内存
    #666
#Xinput=tf.placeholder("float",(None))
Xinput=tf.placeholder("float",(None))
#Yinput=tf.placegolder("float",(None))
Yinput=tf.placeholder("float",(None))

training_epochs=20
with tf.Session() as sess:
    for (x,y),ii in GenerateData(training_epochs):
        xv,yv=sess.run([Xinput,Yinput],feed_dict={Xinput:x,Yinput:y})
            
        print(ii,"|x.shape:",np.shape(xv),"|x[:3]:",xv[:3])
        print(ii,"|y.shape:",np.shape(yv),"|y[:3]:",yv[:3])
            
#显示模拟数据点
train_data=list(GenerateData(1))[0]
plt.plot(train_data[0][0],train_data[0][1],'ro',label='Original data')
plt.legend()
plt.show()   