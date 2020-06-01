# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
def test_numeric_cols_to_bucketized():
    price = tf.feature_column.numeric_column('price')#定义连续数值的特征列

    #将连续数值转成离散值的特征列,离散值共分为3段：小于3、在3与5之间、大于5
    price_bucketized = tf.feature_column.bucketized_column(  price, boundaries=[3.,5.])

    features = {                        #传定义字典
          'price': [[2.], [6.]],
      }

    net = tf.compat.v1.feature_column.input_layer(features,[ price,price_bucketized]) #生成输入层张量
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_bucketized()

def test_numeric_cols_to_identity():
    tf.compat.v1.reset_default_graph()
    price = tf.feature_column.numeric_column('price')#定义连续数值的特征列

    categorical_column = tf.feature_column.categorical_column_with_identity('price', 6)
    print(type(categorical_column))
    one_hot_style = tf.feature_column.indicator_column(categorical_column)
    features = {                        #传定义字典
          'price': [[2], [4]],
      }

    net = tf.compat.v1.feature_column.input_layer(features,[ price,one_hot_style]) #生成输入层张量
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_identity()