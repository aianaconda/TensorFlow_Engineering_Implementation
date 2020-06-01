
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe


tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声

#建立数据集
dataset = tf.data.Dataset.from_tensor_slices( (np.reshape(train_X,[-1,1]),np.reshape(train_X,[-1,1])) )
dataset = dataset.repeat().batch(1)
global_step = tf.train.get_or_create_global_step()
container = tfe.EagerVariableStore()
learning_rate = 0.01
# 随机梯度下降法作为优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

def getcost(x,y):#定义函数，计算loss值
    # 前向结构
    with container.as_default():#将动态图使用的层包装起来。可以得到变量

#        z = tf.contrib.slim.fully_connected(x, 1,reuse=tf.AUTO_REUSE)
        z = tf.layers.dense(x,1, name="l1")
    cost =tf.reduce_mean( tf.square(y - z))#loss值
    return cost

def grad( inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = getcost(inputs, targets)
    return tape.gradient(loss_value,container.trainable_variables())

training_epochs = 20  #迭代训练次数
display_step = 2

#迭代训练模型
for step,value in enumerate(dataset) :
    grads = grad( value[0], value[1])
    optimizer.apply_gradients(zip(grads, container.trainable_variables()), global_step=global_step)
    if step>=training_epochs:
        break

    #显示训练中的详细信息
    if step % display_step == 0:
        cost = getcost (value[0], value[1])
        print ("Epoch:", step+1, "cost=", cost.numpy())

print (" Finished!")
print ("cost=", cost.numpy() )
for i in container.trainable_variables():
    print(i.name,i.numpy())


