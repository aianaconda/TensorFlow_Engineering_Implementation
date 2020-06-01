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
tf.compat.v1.disable_v2_behavior()
#在内存中生成模拟数据
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    return train_X, train_Y   #以生成器的方式返回

train_data = GenerateData()  

batch_size=10

def train_input_fn(train_data, batch_size):  #定义训练数据集输入函数
    #构造数据集的组成：一个特征输入，一个标签输入
    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #将数据集乱序、重复、批次划分. 
    return dataset     #返回数据集 



#定义生成loss可视化的函数
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

tf.compat.v1.reset_default_graph()


features = tf.compat.v1.placeholder("float",[None])#重新定义占位符
labels = tf.compat.v1.placeholder("float",[None])

#其他网络结构不变
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
predictions = tf.multiply(tf.cast(features,dtype = tf.float32), W)+ b# 前向结构
loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=predictions)#定义损失函数

global_step = tf.compat.v1.train.get_or_create_global_step()#重新定义global_step

optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss, global_step=global_step)

saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)#重新定义saver

# 定义学习参数
training_epochs = 500  #设置迭代次数为500
display_step = 2

dataset = train_input_fn(train_data, batch_size)   #重复使用输入函数train_input_fn
one_element = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next() #获得输入数据源张量  

with tf.compat.v1.Session() as sess:

    #恢复估算器检查点文件
    savedir = "myestimatormode/"        
    kpt = tf.train.latest_checkpoint(savedir)       #找到检查点文件
    print("kpt:",kpt)
    saver.restore(sess, kpt)                       #恢复检查点数据
     
    # 向模型输入数据
    while global_step.eval() < training_epochs:
        step = global_step.eval() 
        x,y =sess.run(one_element)

        sess.run(train_op, feed_dict={features: x, labels: y})

        #显示训练中的详细信息
        if step % display_step == 0:
            vloss = sess.run(loss, feed_dict={features: x, labels: y})
            print ("Epoch:", step+1, "cost=", vloss)
            if not (vloss == "NA" ):
                plotdata["batchsize"].append(global_step.eval())
                plotdata["loss"].append(vloss)
            saver.save(sess, savedir+"linermodel.cpkt", global_step)
                
    print (" Finished!")
    saver.save(sess, savedir+"linermodel.cpkt", global_step)
    
    print ("cost=", sess.run(loss,  feed_dict={features: x, labels: y}))


    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()

















