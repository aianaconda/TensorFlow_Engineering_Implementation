# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#演示只有一个连续值特征列的操作
def test_one_column():                      
    price = tf.feature_column.numeric_column('price')          #定义一个特征列

    features = {'price': [[1.], [5.]]}          #将样本数据定义为字典的类型
    net = tf.compat.v1.feature_column.input_layer(features, [price])     #将数据集与特征列一起输入，到input_layer生成张量
    
    with tf.compat.v1.Session() as sess:                  #通过建立会话将其输出
        tt  = sess.run(net)
        print( tt)

test_one_column()

#演示带占位符的特征列操作
def test_placeholder_column():                      
    price = tf.feature_column.numeric_column('price')          #定义一个特征列

    features = {'price':tf.compat.v1.placeholder(dtype=tf.float64)}          #生成一个value为占位符的字典
    net = tf.compat.v1.feature_column.input_layer(features, [price])     #将数据集与特征列一起输入，到input_layer生成张量
    
    with tf.compat.v1.Session() as sess:                  #通过建立会话将其输出
        tt  = sess.run(net, feed_dict={
                features['price']: [[1.], [5.]]
            })
        print( tt)

test_placeholder_column()





import numpy as np
print(np.shape([[[1., 2.]], [[5., 6.]]]))
print(np.shape([[3., 4.], [7., 8.]]))
print(np.shape([[3., 4.]]))
def test_reshaping():
    tf.compat.v1.reset_default_graph()
    price = tf.feature_column.numeric_column('price', shape=[1, 2])#定义一个特征列,并指定形状            
    features = {'price': [[[1., 2.]], [[5., 6.]]]}  #传入一个3维的数组
    features1 = {'price': [[3., 4.], [7., 8.]]}  #传入一个2维的数组

    
    net = tf.compat.v1.feature_column.input_layer(features, price)         #生成特征列张量
    net1 = tf.compat.v1.feature_column.input_layer(features1, price)         #生成特征列张量
    with tf.compat.v1.Session() as sess:                      #通过建立会话将其输出
        print(net.eval())
        print(net1.eval())
        
test_reshaping()

def test_column_order():
    tf.compat.v1.reset_default_graph()
    price_a = tf.feature_column.numeric_column('price_a')   #定义了3个特征列 
    price_b = tf.feature_column.numeric_column('price_b')
    price_c = tf.feature_column.numeric_column('price_c')
    
    features = {                           #创建字典传入数据
          'price_a': [[1.]],
          'price_c': [[4.]],          
          'price_b': [[3.]],
      }
    
    #生成输入层
    net = tf.compat.v1.feature_column.input_layer(features, [price_c, price_a, price_b])
   
    with tf.compat.v1.Session() as sess:             #通过建立会话将其输出
        print(net.eval())

test_column_order()        

