# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf


tf.compat.v1.disable_v2_behavior()
dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )
#dataset1 = tf.data.Dataset.from_tensor_slices( (1,2,3,4,5) )
#dataset1 = tf.data.Dataset.from_tensor_slices( ([1],[2],[3],[4],[5]) )

def getone(dataset):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

one_element1 = getone(dataset1)

with tf.compat.v1.Session() as sess:	# 建立会话（session）
    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element1))				#调用sess.run读出Tensor值




dataset1 = tf.data.Dataset.from_tensor_slices( ([1],[2],[3],[4],[5]) )
one_element1 = getone(dataset1)
with tf.compat.v1.Session() as sess:	# 建立会话（session）
    print(sess.run(one_element1))				#调用sess.run读出Tensor值