# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:39:00 2020
@author: admin
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

def GenerateData(batchsize=100):
    train_x=np.linspace(-1,1,batchsize)
    train_y=2*train_x+np.random.randn(*train_x.shape)*0.3
    yield train_x,train_y

    
Xinput=tf.compat.v1.placeholder("float",(None))
Yinput=tf.compat.v1.placeholder("float",(None))

training_epochs=20
with tf.compat.v1.Session() as sess:
    for epoch in range(training_epochs):
        for x,y in GenerateData():
            xv,yv=sess.run([Xinput,Yinput],feed_dict={Xinput:x,Yinput:y})
            
            print(epoch,"|x.shape:",np.shape(xv),"|x[:3]:",xv[:3])
            print(epoch,"|y.shape:",np.shape(yv),"|y[:3]:",yv[:3])
            
train_data=list(GenerateData())[0]
plt.plot(train_data[0],train_data[1],'ro',label='Original data')
plt.legend()
plt.show()    