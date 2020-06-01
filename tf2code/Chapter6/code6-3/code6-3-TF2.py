# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""



#使用动态图训练一个具有检查点的回归模型

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow.contrib.eager as tfe

#tf.compat.v1.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#（1）生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 定义学习参数
#W = tfe.Variable(tf.random_normal([1]),dtype=tf.float32, name="weight")
#b = tfe.Variable(tf.zeros([1]),dtype=tf.float32, name="bias")
W = tf.Variable(tf.random.normal([1]),dtype=tf.float32, name="weight")
b = tf.Variable(tf.zeros([1]),dtype=tf.float32, name="bias")
global_step = tf.compat.v1.train.get_or_create_global_step()


learning_rate = 0.01
# 随机梯度下降法作为优化器
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
#import numpy as np
#def print_even_callback(op_type, op_name, attrs, inputs, outputs):
#  # A callback that prints only the even output values.
#  print("___",op_type,"___", op_name,"___", attrs, "___",inputs,"___", outputs,"_ok_")
##  print(len(outputs),inputs.numpy())

def getcost(x,y):#定义函数，计算loss值
    # 前向结构
    z = tf.cast(tf.multiply(np.asarray(x,dtype = np.float32), W)+ b,dtype = tf.float32)
#    tfe.add_execution_callback(print_even_callback)
    cost =tf.reduce_mean( input_tensor=tf.square(y - z))#loss值
#    tfe.clear_execution_callbacks()
    return cost

def grad(inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = getcost(inputs, targets)
    return tape.gradient(loss_value,[W,b])



#定义saver，演示两种方法处理检查点文件
savedir = "logeager/"
savedirx = "logeagerx/"
saver = tf.compat.v1.train.Saver([W,b], max_to_keep=1)#生成saver。 max_to_keep=1，表明最多只保存一个检查点文件
#saverx = tfe.Saver([W,b])#生成saver。 max_to_keep=1，表明最多只保存一个检查点文件

kpt = tf.train.latest_checkpoint(savedir)#找到检查点文件
kptx = tf.train.latest_checkpoint(savedirx)#找到检查点文件
if kpt!=None:
    saver.restore(None, kpt) #两种加载方式都可以
    #saverx.restore(kptx)

training_epochs = 20  #迭代训练次数
display_step = 2

plotdata = { "batchsize":[], "loss":[] }#收集训练参数

while global_step/len(train_X) < training_epochs: #迭代训练模型
    step = int( global_step/len(train_X) )
    for (x, y) in zip(train_X, train_Y):
        grads = grad( x, y)
        optimizer.apply_gradients(zip(grads, [W,b]), global_step=global_step)



    #显示训练中的详细信息
    if step % display_step == 0:
        cost = getcost (x, y).numpy()
        print ("Epoch:", step+1, "cost=", cost,"W=", W.numpy(), "b=", b.numpy())
        if not (cost == "NA" ):
            plotdata["batchsize"].append(global_step.numpy())
            plotdata["loss"].append(cost)
        saver.save(None, savedir+"linermodel.cpkt", global_step)
        #saverx.save(savedirx+"linermodel.cpkt", global_step)


print (" Finished!")
saver.save(None, savedir+"linermodel.cpkt", global_step)
#saverx.save(savedirx+"linermodel.cpkt", global_step)
print ("cost=", getcost (train_X, train_Y).numpy() , "W=", W.numpy(), "b=", b.numpy())

#显示模型
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, W * train_X + b, label='Fitted line')
plt.legend()
plt.show()

def moving_average(a, w=10):#定义生成loss可视化的函数
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

plotdata["avgloss"] = moving_average(plotdata["loss"])
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()
