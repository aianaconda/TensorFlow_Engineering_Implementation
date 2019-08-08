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

#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

#定义生成loss可视化的函数
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

tf.reset_default_graph()

# 定义学习参数
training_epochs = 58  #设置迭代次数为58
display_step = 2

with tf.Session() as sess:
    savedir = "log2/"
    kpt = tf.train.latest_checkpoint(savedir)       #找到检查点文件
    print("kpt:",kpt)
    new_saver = tf.train.import_meta_graph(kpt+'.meta')  #从检查点的meta文件中导入变量
    new_saver.restore(sess, kpt)                            #恢复检查点数据
    
        
    print(tf.get_collection('optimizer'))               #通过集合取张量
    optimizer = tf.get_collection('optimizer')[0]#返回的是一个list ，只是取第一个
    X=tf.get_collection('X')[0]
    Y=tf.get_collection('Y')[0]
    cost=tf.get_collection('cost')[0]
    result=tf.get_collection('result')[0]
    global_step = tf.get_collection('global_step')[0]
     
    #节点恢复完成，可以继续训练
    while global_step.eval()/len(train_X) < training_epochs:
        step = int( global_step.eval()/len(train_X) )
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", step+1, "cost=", loss)
            if not (loss == "NA" ):
                plotdata["batchsize"].append(global_step.eval())
                plotdata["loss"].append(loss)
            new_saver.save(sess, savedir+"linermodel.cpkt", global_step)
                
    print (" Finished!")
    new_saver.save(sess, savedir+"linermodel.cpkt", global_step)
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}))

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()
