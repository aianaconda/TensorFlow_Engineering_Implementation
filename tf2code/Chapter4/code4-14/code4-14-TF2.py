# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf


tf.compat.v1.disable_v2_behavior()

dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )	#定义训练数据集

#创建迭代器
iterator1 = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(dataset1),tf.compat.v1.data.get_output_shapes(dataset1))

one_element1 = iterator1.get_next()

with tf.compat.v1.Session()  as sess2:
    sess2.run( iterator1.make_initializer(dataset1) )#初始化迭代器
    for ii in range(2):  #数据集迭代两次
        while True:		#通过for循环打印所有的数据
            try:
                print(sess2.run(one_element1))				#调用sess.run读出Tensor值
            except tf.errors.OutOfRangeError:
                print("遍历结束")
                sess2.run( iterator1.make_initializer(dataset1) )# 从头再来一遍
                break


    print(sess2.run(one_element1,{one_element1:356}))  #往数据集中注入数据


dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )	#定义训练数据集
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset1)  #生成一个迭代器

dataset_test = tf.data.Dataset.from_tensor_slices( [10,20,30,40,50] )#定义测试数据集
iterator_test = tf.compat.v1.data.make_one_shot_iterator(dataset1)  #生成一个迭代器
#适用于测试与训练场景下的数据集方式
with tf.compat.v1.Session()  as sess:
    iterator_handle = sess.run(iterator.string_handle())
    iterator_handle_test = sess.run(iterator_test.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator3 = tf.compat.v1.data.Iterator.from_string_handle(handle, iterator.output_types)

    one_element3 = iterator3.get_next()
    print(sess.run(one_element3,{handle: iterator_handle}))
    print(sess.run(one_element3,{handle: iterator_handle_test}))














