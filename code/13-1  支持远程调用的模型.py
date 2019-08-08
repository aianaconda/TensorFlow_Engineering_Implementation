# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
#使用静态图训练一个具有检查点的回归模型

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#（1）生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

#（2）建立网络模型

# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 前向结构
z = tf.multiply(X, W)+ b
global_step = tf.Variable(0, name='global_step', trainable=False)
#反向优化
cost =tf.reduce_mean( tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step) #梯度下降

# 初始化所有变量
init = tf.global_variables_initializer()
# 定义学习参数
training_epochs = 28
display_step = 2

savedir = "log/"
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)#生成saver。 max_to_keep=1，表明最多只保存一个检查点文件

#定义生成loss可视化的函数
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#（3）建立session进行训练
with tf.Session() as sess:
    sess.run(init)
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt!=None:
        saver.restore(sess, kpt)
     
    # 向模型输入数据
    while global_step.eval()/len(train_X) < training_epochs:
        step = int( global_step.eval()/len(train_X) )
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", step+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(global_step.eval())
                plotdata["loss"].append(loss)
            saver.save(sess, savedir+"linermodel.cpkt", global_step)
                
    print (" Finished!")
    saver.save(sess, savedir+"linermodel.cpkt", global_step)
    
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    #显示模型
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()


   
#################导出模型#######################
    from tensorflow.python.saved_model import tag_constants
    builder = tf.saved_model.builder.SavedModelBuilder(savedir+'tfservingmodelv1')
    
    #定义输入签名，X为输入tensor
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(X)}
    #定义输出签名， z为最终需要的输出结果tensor 
    outputs = {'output' : tf.saved_model.utils.build_tensor_info(z)}
    signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    
    #将节点的定义和值加到builder中，同时还加入了标签， 还可以使用TRAINING、GPU或自定义
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'my_signature':signature})
    builder.save()     


    

    
    